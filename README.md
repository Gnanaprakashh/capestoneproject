# Telangana Travel Guide

A Flask travel-planning MVP for Telangana. It keeps the existing Random Forest recommendations and local place dataset, and adds Explore, transport search links, hotels, itinerary and budget planning, accounts, and saved trips.

## Run locally

Use Python 3.12 or newer and PowerShell:

```powershell
Set-Location 'C:\path\to\project trails'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:SECRET_KEY = (& python -c "import secrets; print(secrets.token_hex(32))")
$env:FLASK_DEBUG = "true"
python app.py
```

Open <http://127.0.0.1:5000>. Without database environment variables, the app uses SQLite at `instance/travel.db`. Set `SECRET_KEY` to a stable, private value so sessions remain valid after restarting the app.

## Run with MySQL and Docker

Copy `.env.example` to `.env`, replace every sample value with unique private values, and run:

```powershell
docker compose up --build
```

Open <http://127.0.0.1:8000>. Docker Compose waits for MySQL before starting the application. Never commit `.env`; the database volume keeps account and trip data between restarts.

For an existing MySQL deployment, set `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`, and `SECRET_KEY`. Alternatively, set `DATABASE_URL` to a SQLAlchemy URL such as `mysql+pymysql://user:password@host:3306/database?charset=utf8mb4`.

## Page and integration boundaries

- Flights, trains, buses, and hotel availability open external provider searches. The application does not retrieve live fares, compare actual inventory, create reservations, or process payments.
- Budget totals use traveler-provided sample cost assumptions. They are estimates, not provider quotes; transport and taxes are not included.
- The travel assistant uses the local dataset and deterministic planning logic. It does not call an LLM.
- Road routes use the public OSRM demo service and require an internet connection.
- Account passwords are stored as password hashes. Saved itineraries are private to the signed-in user. Configure a stable `SECRET_KEY` for deployment and serve production traffic over HTTPS.
