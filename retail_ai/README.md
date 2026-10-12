# AI-Based Retail Management System

A simple base project: inventory + point-of-sale + three lightweight AI features.

## Features
- **Inventory**: add products, view stock levels
- **Sales**: record a sale; stock is reduced automatically
- **AI demand forecasting**: linear-regression trend on the last 30 days of sales, predicts the next 7 days
- **AI smart reorder**: suggests how many units to order (forecast + safety stock - current stock)
- **AI recommendations**: "customers also bought" using co-purchase counts

Demo data (8 products, 60 days of orders) is generated on first run.

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## JSON API
- `GET /api/forecast` - forecast and reorder suggestion for every product
- `GET /api/recommend/<product_id>` - recommended products

## Project structure
- `app.py` - database, AI logic, routes and HTML template in one file
- `retail.db` - SQLite database (created automatically; delete it to reset)

## Ideas for extending
- Swap the linear model for scikit-learn / Prophet
- Add user login, suppliers, and purchase orders
- Add charts (Chart.js) for sales history
