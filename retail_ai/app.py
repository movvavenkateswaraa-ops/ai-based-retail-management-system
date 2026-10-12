"""AI-Based Retail Management System (simple base project).

Features
- Product inventory management (add products, view stock)
- Point-of-sale (record a sale, stock updates automatically)
- AI 1: Demand forecasting  - linear-regression trend on the last 30 days of sales
- AI 2: Smart reorder       - suggests how many units to order for the next 7 days
- AI 3: Recommendations     - "customers also bought" via co-purchase counting

Run:  pip install -r requirements.txt  &&  python app.py
Open: http://127.0.0.1:5000
"""
import os
import random
import sqlite3
from collections import Counter
from datetime import date, timedelta

from flask import Flask, g, jsonify, redirect, render_template_string, request, url_for

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail.db")
HISTORY_DAYS = 30
FORECAST_DAYS = 7

app = Flask(__name__)


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and fill them with demo data on first run."""
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL,
            reorder_level INTEGER NOT NULL DEFAULT 10
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity INTEGER NOT NULL
        );
        """
    )
    if db.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
        seed(db)
    db.commit()
    db.close()


def seed(db):
    random.seed(42)
    products = [
        ("Milk 1L", "Dairy", 1.2),
        ("Bread", "Bakery", 1.0),
        ("Eggs (12)", "Dairy", 2.5),
        ("Butter", "Dairy", 3.0),
        ("Coffee 250g", "Beverages", 6.5),
        ("Tea Box", "Beverages", 4.0),
        ("Rice 5kg", "Grocery", 9.0),
        ("Cooking Oil 1L", "Grocery", 5.5),
    ]
    for name, cat, price in products:
        db.execute(
            "INSERT INTO products (name, category, price, stock, reorder_level) VALUES (?,?,?,?,?)",
            (name, cat, price, random.randint(15, 60), 10),
        )
    # Frequently bought together pairs (gives the recommender something to learn)
    bundles = [(1, 2), (1, 3), (5, 6), (2, 4), (7, 8)]
    today = date.today()
    for d in range(60, 0, -1):
        day = (today - timedelta(days=d)).isoformat()
        for _ in range(random.randint(5, 12)):
            cur = db.execute("INSERT INTO orders (order_date) VALUES (?)", (day,))
            if random.random() < 0.5:
                items = set(random.choice(bundles))
            else:
                items = set(random.sample(range(1, len(products) + 1), random.randint(1, 3)))
            for pid in items:
                db.execute(
                    "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?,?,?)",
                    (cur.lastrowid, pid, random.randint(1, 3)),
                )


# --------------------------------------------------------------------------
# AI features
# --------------------------------------------------------------------------
def daily_sales(db, product_id, days=HISTORY_DAYS):
    """Units sold per day for the last `days` days (zeros filled in)."""
    start = date.today() - timedelta(days=days)
    rows = db.execute(
        """SELECT o.order_date AS d, SUM(oi.quantity) AS q
           FROM order_items oi JOIN orders o ON o.id = oi.order_id
           WHERE oi.product_id = ? AND o.order_date >= ?
           GROUP BY o.order_date""",
        (product_id, start.isoformat()),
    ).fetchall()
    by_day = {r["d"]: r["q"] for r in rows}
    return [by_day.get((start + timedelta(days=i)).isoformat(), 0) for i in range(days)]


def forecast_demand(history, horizon=FORECAST_DAYS):
    """Fit y = a + b*x by least squares and predict the next `horizon` days."""
    n = len(history)
    if n == 0:
        return [0.0] * horizon
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(history) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, history)) / var_x
        if var_x
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    return [max(0.0, intercept + slope * (n + i)) for i in range(horizon)]


def analyse_product(db, product):
    history = daily_sales(db, product["id"])
    preds = forecast_demand(history)
    expected = round(sum(preds))
    # Safety stock = reorder level; order enough to cover the forecast + safety.
    needed = expected + product["reorder_level"] - product["stock"]
    reorder_qty = max(0, needed)
    if product["stock"] <= product["reorder_level"]:
        status = "LOW STOCK"
    elif reorder_qty > 0:
        status = "REORDER SOON"
    else:
        status = "OK"
    return {
        "product_id": product["id"],
        "name": product["name"],
        "stock": product["stock"],
        "avg_daily_sales": round(sum(history) / len(history), 2),
        "forecast_next_7_days": expected,
        "suggested_reorder_qty": reorder_qty,
        "status": status,
    }


def recommend(db, product_id, top_n=3):
    """Customers who bought this product also bought... (co-purchase counts)."""
    order_ids = [
        r["order_id"]
        for r in db.execute(
            "SELECT DISTINCT order_id FROM order_items WHERE product_id = ?", (product_id,)
        )
    ]
    if not order_ids:
        return []
    marks = ",".join("?" * len(order_ids))
    rows = db.execute(
        f"""SELECT product_id FROM order_items
            WHERE order_id IN ({marks}) AND product_id != ?""",
        (*order_ids, product_id),
    ).fetchall()
    counts = Counter(r["product_id"] for r in rows)
    result = []
    for pid, cnt in counts.most_common(top_n):
        name = db.execute("SELECT name FROM products WHERE id = ?", (pid,)).fetchone()["name"]
        result.append({"product_id": pid, "name": name, "co_purchases": cnt})
    return result


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------
PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Retail Management System</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #222; }
  header { background: #1f3a5f; color: #fff; padding: 16px 24px; }
  main { max-width: 1100px; margin: 20px auto; padding: 0 16px; }
  section { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  h2 { margin-top: 0; font-size: 18px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; }
  th { background: #f9fafb; }
  form { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  input, select, button { padding: 7px 10px; font-size: 14px; }
  button { background: #1f3a5f; color: #fff; border: 0; border-radius: 4px; cursor: pointer; }
  .tag { padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #fff; }
  .LOW { background: #d93025; } .REORDER { background: #f29900; } .OK { background: #188038; }
  .msg { padding: 8px 12px; background: #e8f0fe; border-radius: 4px; margin-bottom: 16px; }
</style>
</head>
<body>
<header><h1 style="margin:0;font-size:22px">AI-Based Retail Management System</h1></header>
<main>
  {% if msg %}<div class="msg">{{ msg }}</div>{% endif %}

  <section>
    <h2>AI Insights: Demand Forecast &amp; Reorder Suggestions</h2>
    <table>
      <tr><th>Product</th><th>Stock</th><th>Avg daily sales</th>
          <th>Forecast (next 7 days)</th><th>Suggested reorder</th><th>Status</th>
          <th>Customers also buy</th></tr>
      {% for a in analysis %}
      <tr>
        <td>{{ a.name }}</td><td>{{ a.stock }}</td><td>{{ a.avg_daily_sales }}</td>
        <td>{{ a.forecast_next_7_days }}</td><td>{{ a.suggested_reorder_qty }}</td>
        <td><span class="tag {{ a.status.split()[0] }}">{{ a.status }}</span></td>
        <td>{{ recs[a.product_id] | map(attribute='name') | join(', ') }}</td>
      </tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Record a Sale</h2>
    <form method="post" action="{{ url_for('sell') }}">
      <select name="product_id" required>
        {% for p in products %}<option value="{{ p.id }}">{{ p.name }} (stock {{ p.stock }})</option>{% endfor %}
      </select>
      <input type="number" name="quantity" value="1" min="1" required>
      <button>Sell</button>
    </form>
  </section>

  <section>
    <h2>Add Product</h2>
    <form method="post" action="{{ url_for('add_product') }}">
      <input name="name" placeholder="Name" required>
      <input name="category" placeholder="Category" required>
      <input type="number" step="0.01" name="price" placeholder="Price" required>
      <input type="number" name="stock" placeholder="Stock" required>
      <input type="number" name="reorder_level" placeholder="Reorder level" value="10" required>
      <button>Add</button>
    </form>
  </section>

  <section>
    <h2>Inventory</h2>
    <table>
      <tr><th>ID</th><th>Name</th><th>Category</th><th>Price</th><th>Stock</th><th>Reorder level</th></tr>
      {% for p in products %}
      <tr><td>{{ p.id }}</td><td>{{ p.name }}</td><td>{{ p.category }}</td>
          <td>{{ "%.2f"|format(p.price) }}</td><td>{{ p.stock }}</td><td>{{ p.reorder_level }}</td></tr>
      {% endfor %}
    </table>
  </section>
</main>
</body>
</html>
"""


@app.route("/")
def dashboard():
    db = get_db()
    products = db.execute("SELECT * FROM products ORDER BY id").fetchall()
    analysis = [analyse_product(db, p) for p in products]
    recs = {p["id"]: recommend(db, p["id"]) for p in products}
    return render_template_string(
        PAGE, products=products, analysis=analysis, recs=recs, msg=request.args.get("msg")
    )


@app.route("/products", methods=["POST"])
def add_product():
    db = get_db()
    try:
        db.execute(
            "INSERT INTO products (name, category, price, stock, reorder_level) VALUES (?,?,?,?,?)",
            (
                request.form["name"].strip(),
                request.form["category"].strip(),
                float(request.form["price"]),
                int(request.form["stock"]),
                int(request.form["reorder_level"]),
            ),
        )
        db.commit()
        msg = "Product added."
    except (sqlite3.IntegrityError, ValueError):
        msg = "Could not add product (duplicate name or invalid values)."
    return redirect(url_for("dashboard", msg=msg))


@app.route("/sell", methods=["POST"])
def sell():
    db = get_db()
    pid = int(request.form["product_id"])
    qty = int(request.form["quantity"])
    product = db.execute("SELECT * FROM products WHERE id = ?", (pid,)).fetchone()
    if product is None or qty < 1:
        msg = "Invalid sale."
    elif product["stock"] < qty:
        msg = f"Not enough stock for {product['name']} (only {product['stock']} left)."
    else:
        cur = db.execute("INSERT INTO orders (order_date) VALUES (?)", (date.today().isoformat(),))
        db.execute(
            "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?,?,?)",
            (cur.lastrowid, pid, qty),
        )
        db.execute("UPDATE products SET stock = stock - ? WHERE id = ?", (qty, pid))
        db.commit()
        msg = f"Sold {qty} x {product['name']}."
    return redirect(url_for("dashboard", msg=msg))


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------
@app.route("/api/forecast")
def api_forecast():
    db = get_db()
    products = db.execute("SELECT * FROM products").fetchall()
    return jsonify([analyse_product(db, p) for p in products])


@app.route("/api/recommend/<int:product_id>")
def api_recommend(product_id):
    return jsonify(recommend(get_db(), product_id))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
