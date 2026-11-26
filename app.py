import json
from itertools import combinations
import os
from pathlib import Path
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
import joblib
import numpy as np
import pandas as pd
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from extensions import db, login_manager
from models import Trip, User


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
database_url = os.environ.get("DATABASE_URL")
if not database_url and os.environ.get("MYSQL_HOST"):
    mysql_user = quote_plus(os.environ["MYSQL_USER"])
    mysql_password = quote_plus(os.environ["MYSQL_PASSWORD"])
    mysql_host = os.environ["MYSQL_HOST"]
    mysql_port = os.environ.get("MYSQL_PORT", "3306")
    mysql_database = quote_plus(os.environ["MYSQL_DATABASE"])
    database_url = (
        f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:"
        f"{mysql_port}/{mysql_database}?charset=utf8mb4"
    )
app.config["SQLALCHEMY_DATABASE_URI"] = database_url or (
    f"sqlite:///{(BASE_DIR / 'instance' / 'travel.db').as_posix()}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true"
app.config["WTF_CSRF_TIME_LIMIT"] = 3600
Path(app.instance_path).mkdir(parents=True, exist_ok=True)
db.init_app(app)
login_manager.init_app(app)
csrf = CSRFProtect(app)
ROUTE_CORRIDOR_KM = 15.0
MAX_ITINERARY_STOPS = 5
SERVICE_CATEGORIES = {"Hotel", "Restaurant", "Petrol Station"}
DEFAULT_HOTEL_ESTIMATE = 2500
DEFAULT_MEALS_ESTIMATE = 600
DEFAULT_ACTIVITY_ESTIMATE = 300

df = pd.read_csv(BASE_DIR / "combined_telangana_places.csv")
model = joblib.load(BASE_DIR / "combined_places_model.pkl")
le = joblib.load(BASE_DIR / "category_encoder.pkl")
location_names = list(df["Place"].unique())
categories = sorted(df["Category"].dropna().unique())
default_interests = [category for category in categories if category not in SERVICE_CATEGORIES]


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


@app.context_processor
def inject_navigation():
    return {"navigation_user": current_user}


class RoutingError(Exception):
    """Raised when the road-routing service cannot return a route."""


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def get_road_route(start_coords, dest_coords):
    """Fetch a driving route and its road geometry from the public OSRM demo service."""
    start_lat, start_lon = start_coords
    dest_lat, dest_lon = dest_coords
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{dest_lon},{dest_lat}"
        "?overview=full&geometries=geojson&steps=false"
    )
    req = Request(url, headers={"User-Agent": "TelanganaTravelGuide/1.0"})
    try:
        with urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RoutingError("The road-routing service is unavailable. Please try again shortly.") from exc

    routes = result.get("routes", [])
    if result.get("code") != "Ok" or not routes:
        raise RoutingError("No driving route was found between those locations.")

    route = routes[0]
    coordinates = route.get("geometry", {}).get("coordinates", [])
    if len(coordinates) < 2:
        raise RoutingError("The road-routing service returned an incomplete route.")

    return {
        "coordinates": coordinates,
        "distance_km": route["distance"] / 1000,
        "duration_minutes": route["duration"] / 60,
    }


def measure_places_against_route(places, route_coordinates):
    """Return each place's closest distance to, and progress along, the road route."""
    earth_radius_km = 6371.0
    route_points = [(float(lat), float(lon)) for lon, lat in route_coordinates]
    segment_lengths = [
        haversine(lat1, lon1, lat2, lon2)
        for (lat1, lon1), (lat2, lon2) in zip(route_points, route_points[1:])
    ]
    cumulative_km = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    route_distances = []
    route_progress = []
    for _, place in places.iterrows():
        place_lat = float(place["Latitude"])
        place_lon = float(place["Longitude"])
        nearest_distance = float("inf")
        nearest_progress = 0.0

        for index, ((lat1, lon1), (lat2, lon2)) in enumerate(
            zip(route_points, route_points[1:])
        ):
            mean_lat = np.radians((place_lat + lat1 + lat2) / 3)
            start_x = earth_radius_km * np.radians(lon1 - place_lon) * np.cos(mean_lat)
            start_y = earth_radius_km * np.radians(lat1 - place_lat)
            end_x = earth_radius_km * np.radians(lon2 - place_lon) * np.cos(mean_lat)
            end_y = earth_radius_km * np.radians(lat2 - place_lat)
            segment_x = end_x - start_x
            segment_y = end_y - start_y
            segment_squared = segment_x * segment_x + segment_y * segment_y
            fraction = 0.0 if segment_squared == 0 else (
                -(start_x * segment_x + start_y * segment_y) / segment_squared
            )
            fraction = min(1.0, max(0.0, fraction))
            offset_x = start_x + fraction * segment_x
            offset_y = start_y + fraction * segment_y
            distance = float(np.hypot(offset_x, offset_y))

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_progress = cumulative_km[index] + fraction * segment_lengths[index]

        route_distances.append(nearest_distance)
        route_progress.append(nearest_progress)

    result = places.copy()
    result["Route_Distance_km"] = route_distances
    result["Along_Route_km"] = route_progress
    return result


def build_recommendations(route_places, start_coords, dest_coords, interests):
    start_distances = [
        haversine(start_coords[0], start_coords[1], row["Latitude"], row["Longitude"])
        for _, row in route_places.iterrows()
    ]
    dest_distances = [
        haversine(dest_coords[0], dest_coords[1], row["Latitude"], row["Longitude"])
        for _, row in route_places.iterrows()
    ]
    category_encoded = le.transform(route_places["Category"])
    features = pd.DataFrame(
        {
            "Distance_Start": start_distances,
            "Distance_Dest": dest_distances,
            "Category_Encoded": category_encoded,
        }
    )
    route_places = route_places.copy()
    route_places["Predicted_Score"] = model.predict(features)

    score_range = route_places["Predicted_Score"].max() - route_places["Predicted_Score"].min()
    if score_range == 0:
        model_score = pd.Series(0.5, index=route_places.index)
    else:
        model_score = (
            (route_places["Predicted_Score"] - route_places["Predicted_Score"].min())
            / score_range
        )
    route_fit = 1 - route_places["Route_Distance_km"].clip(upper=ROUTE_CORRIDOR_KM) / ROUTE_CORRIDOR_KM
    if interests:
        preference_score = route_places["Category"].isin(interests).astype(float)
        route_places["Personalized_Score"] = (
            0.5 * model_score + 0.25 * route_fit + 0.25 * preference_score
        )
    else:
        route_places["Personalized_Score"] = 0.67 * model_score + 0.33 * route_fit

    route_places["Distance_Start_km"] = np.round(start_distances, 2)
    route_places["Distance_Dest_km"] = np.round(dest_distances, 2)
    route_places["Route_Distance_km"] = route_places["Route_Distance_km"].round(2)
    route_places["Along_Route_km"] = route_places["Along_Route_km"].round(1)
    return route_places


def build_itinerary(recommendations, start_name, dest_name):
    tourist_places = recommendations[
        ~recommendations["Category"].isin(SERVICE_CATEGORIES)
        & ~recommendations["Place"].isin([start_name, dest_name])
    ]
    selected = tourist_places.nlargest(MAX_ITINERARY_STOPS, "Personalized_Score")
    itinerary = selected.sort_values("Along_Route_km").copy()
    previous_progress = 0.0
    stop_numbers = []
    leg_distances = []
    for stop_number, (_, place) in enumerate(itinerary.iterrows(), start=1):
        stop_numbers.append(stop_number)
        leg_distances.append(round(max(0.0, place["Along_Route_km"] - previous_progress), 1))
        previous_progress = place["Along_Route_km"]
    itinerary["Stop_Number"] = stop_numbers
    itinerary["Leg_Distance_km"] = leg_distances
    return itinerary


def build_budget_plan(recommendations, itinerary, days, travelers, budget,
                      hotel_per_night, meals_per_person_day, activity_per_stop):
    """Choose an affordable combination of stops and stay estimates, not live quotes."""
    days = max(1, days)
    travelers = max(1, travelers)
    nights = max(0, days - 1)
    hotels = recommendations[recommendations["Category"] == "Hotel"].copy()
    if not hotels.empty:
        hotels["Estimated_Nightly_Cost"] = hotels["Description"].str.lower().apply(
            lambda description: 6000 if "luxury" in description or "5-star" in description
            else 4500 if "premium" in description
            else 3000 if "3-star" in description
            else 2000 if "haritha" in description or "government-run" in description
            else hotel_per_night
        )
    meal_cost = meals_per_person_day * travelers * days
    activity_count = len(itinerary)
    hotel_records = hotels.to_dict("records")
    for hotel in (hotel_records if nights else []):
        nightly_cost = int(hotel["Estimated_Nightly_Cost"])
        hotel["Estimated_Nightly_Cost"] = nightly_cost

    affordable_hotels = [
        hotel for hotel in hotel_records
        if hotel["Estimated_Nightly_Cost"] <= hotel_per_night
    ]
    stay_choices = (
        [(hotel, hotel["Estimated_Nightly_Cost"] * nights) for hotel in affordable_hotels]
        if nights and affordable_hotels
        else [(None, hotel_per_night * nights)]
    )
    itinerary = itinerary.sort_values("Along_Route_km").copy()
    evaluated_plans = []
    for stop_count in range(len(itinerary) + 1):
        for stop_indices in combinations(range(len(itinerary)), stop_count):
            stops = itinerary.iloc[list(stop_indices)]
            stop_cost = activity_per_stop * len(stops)
            stop_score = float(stops["Personalized_Score"].sum())
            for hotel, stay_cost in stay_choices:
                total_cost = stay_cost + meal_cost + stop_cost
                evaluated_plans.append(
                    {
                        "indices": stop_indices,
                        "hotel": hotel,
                        "hotel_cost": stay_cost,
                        "activity_cost": stop_cost,
                        "total_cost": total_cost,
                        "score": stop_score,
                        "within_budget": total_cost <= budget,
                    }
                )
    chosen_plan = min(
        evaluated_plans,
        key=lambda plan: (
            not plan["within_budget"],
            max(0, plan["total_cost"] - budget),
            -plan["score"],
            -len(plan["indices"]),
            plan["total_cost"],
        ),
    )
    itinerary = itinerary.iloc[list(chosen_plan["indices"])].copy()
    itinerary["Day"] = [
        min(days, (number - 1) // max(1, (len(itinerary) + days - 1) // days) + 1)
        for number in range(1, len(itinerary) + 1)
    ]
    previous_progress = 0.0
    leg_distances = []
    for progress in itinerary["Along_Route_km"]:
        leg_distances.append(round(max(0.0, progress - previous_progress), 1))
        previous_progress = progress
    itinerary["Leg_Distance_km"] = leg_distances

    hotel_options = []
    for hotel in hotel_records:
        stay_cost = hotel["Estimated_Nightly_Cost"] * nights
        trip_cost = stay_cost + meal_cost + chosen_plan["activity_cost"]
        hotel_options.append(
            {
                **hotel,
                "Estimated_Stay_Cost": stay_cost,
                "Estimated_Trip_Cost": trip_cost,
                "Within_Budget": trip_cost <= budget,
                "Within_Nightly_Limit": hotel["Estimated_Nightly_Cost"] <= hotel_per_night,
            }
        )
    hotel_options.sort(
        key=lambda option: (
            not (option["Within_Budget"] and option["Within_Nightly_Limit"]),
            option["Estimated_Trip_Cost"],
            -option["Personalized_Score"],
        )
    )
    hotel_cost = chosen_plan["hotel_cost"]
    return {
        "days": days,
        "travelers": travelers,
        "budget": budget,
        "hotel_per_night": hotel_per_night,
        "meals_per_person_day": meals_per_person_day,
        "activity_per_stop": activity_per_stop,
        "hotel_cost": hotel_cost,
        "meal_cost": meal_cost,
        "activity_cost": chosen_plan["activity_cost"],
        "total_cost": chosen_plan["total_cost"],
        "remaining": budget - chosen_plan["total_cost"],
        "within_budget": chosen_plan["within_budget"],
        "hotel_options": hotel_options[:3],
        "selected_hotel": chosen_plan["hotel"] if chosen_plan["within_budget"] else None,
        "optimized_stops": len(itinerary),
        "original_stops": activity_count,
        "itinerary": itinerary,
    }


def answer_travel_question(question):
    """Answer common travel questions from the local Telangana places dataset."""
    text = question.strip()
    if not text:
        return "Ask me about places, or try: Plan a 3-day trip to Warangal under ₹5,000."

    days_match = re.search(r"\b(\d{1,2})\s*[- ]?\s*days?\b", text, re.IGNORECASE)
    days = min(14, max(1, int(days_match.group(1)))) if days_match else 1
    budget_match = re.search(
        r"(?:under|below|within|budget(?:\s+of)?|₹|rs\.?|inr)\s*₹?\s*([\d,]+)",
        text,
        re.IGNORECASE,
    )
    budget = int(budget_match.group(1).replace(",", "")) if budget_match else None
    lowered = text.casefold()

    interest_terms = {
        "waterfall": ("waterfall", "Nature"),
        "nature": ("", "Nature"),
        "temple": ("temple", "Religious"),
        "histor": ("", "Historical"),
        "fort": ("fort", "Historical"),
        "lake": ("lake", "Lake"),
        "wildlife": ("", "Wildlife"),
        "adventure": ("hill", "Nature"),
        "culture": ("", "Cultural"),
    }
    search_term = None
    preferred_category = None
    for term, (keyword, category) in interest_terms.items():
        if term in lowered:
            search_term = keyword or None
            preferred_category = category
            break

    location_terms = set()
    for place in df.itertuples():
        candidates = [place.Place]
        candidates.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", place.Description))
        for candidate in candidates:
            normalized = candidate.casefold()
            if len(normalized) > 3 and normalized in lowered:
                location_terms.add(normalized)

    relevant = df.copy()
    if location_terms:
        location_mask = relevant["Place"].str.casefold().apply(
            lambda value: any(term in value for term in location_terms)
        ) | relevant["Description"].str.casefold().apply(
            lambda value: any(term in value for term in location_terms)
        )
        relevant = relevant[location_mask]
    if search_term:
        relevant = relevant[
            relevant["Place"].str.contains(search_term, case=False, na=False)
            | relevant["Description"].str.contains(search_term, case=False, na=False)
            | (relevant["Category"] == preferred_category)
        ]
    elif preferred_category:
        relevant = relevant[relevant["Category"] == preferred_category]

    attractions = relevant[~relevant["Category"].isin(SERVICE_CATEGORIES)].drop_duplicates("Place")
    if attractions.empty:
        return (
            "I couldn’t find matching places in the included Telangana dataset. "
            "Try a place such as Warangal, Yadadri, Medak, or a category such as temples or nature."
        )

    places_per_day = max(1, min(2, (MAX_ITINERARY_STOPS + days - 1) // days))
    selected = attractions.head(min(MAX_ITINERARY_STOPS, days * places_per_day))
    lines = []
    if location_terms:
        lines.append(f"Here are ideas matching your question about {', '.join(sorted(location_terms))}:")
    else:
        lines.append("Here are ideas from the Telangana places in this app:")
    for index, (_, place) in enumerate(selected.iterrows()):
        day = min(days, index // places_per_day + 1)
        lines.append(f"Day {day}: {place['Place']} ({place['Category']}) — {place['Description']}")

    if budget is not None:
        estimated_total = (
            DEFAULT_HOTEL_ESTIMATE * max(0, days - 1)
            + DEFAULT_MEALS_ESTIMATE * days
            + DEFAULT_ACTIVITY_ESTIMATE * len(selected)
        )
        lines.append(
            f"Rough sample estimate for 1 traveler: ₹{estimated_total:,} "
            f"(hotel ₹{DEFAULT_HOTEL_ESTIMATE:,}/night, meals ₹{DEFAULT_MEALS_ESTIMATE:,}/day, "
            f"activities ₹{DEFAULT_ACTIVITY_ESTIMATE:,}/stop). "
            f"This {'fits' if estimated_total <= budget else 'is over'} your ₹{budget:,} budget. "
            "These are editable planning assumptions, not current prices; transport is not included."
        )

    matching_services = relevant[relevant["Category"].isin({"Hotel", "Restaurant"})]
    for category, label in (("Hotel", "Nearby stays to check"), ("Restaurant", "Food options to check")):
        options = matching_services[matching_services["Category"] == category]["Place"].head(3).tolist()
        if options:
            lines.append(f"{label}: {', '.join(options)}.")
    lines.append("For route-based stops and editable trip costs, use the trip planner above.")
    return "\n".join(lines)


@app.get("/")
def home():
    return render_template("home.html", page_title="Home")


@app.get("/explore")
def explore():
    search_text = request.args.get("q", "").strip()
    category = request.args.get("category", "")
    places = df.copy()
    if category in categories:
        places = places[places["Category"] == category]
    if search_text:
        mask = places["Place"].str.contains(search_text, case=False, na=False) | places[
            "Description"
        ].str.contains(search_text, case=False, na=False)
        places = places[mask]
    return render_template(
        "explore.html", page_title="Explore Telangana", places=places,
        categories=categories, search_text=search_text, selected_category=category,
    )


def build_external_searches(kind, origin, destination, travel_date):
    query = f"{kind} from {origin} to {destination}"
    if travel_date:
        query += f" on {travel_date}"
    providers = {
        "flights": [
            ("Google Flights", "https://www.google.com/travel/flights?" + urlencode({"q": query})),
            ("MakeMyTrip search", "https://www.google.com/search?" + urlencode({"q": f"site:makemytrip.com/flights {query}"})),
            ("Skyscanner search", "https://www.google.com/search?" + urlencode({"q": f"site:skyscanner.co.in {query}"})),
        ],
        "trains": [
            ("IRCTC search", "https://www.google.com/search?" + urlencode({"q": f"site:irctc.co.in {query}"})),
            ("ixigo trains", "https://www.google.com/search?" + urlencode({"q": f"site:ixigo.com/trains {query}"})),
        ],
        "buses": [
            ("redBus search", "https://www.google.com/search?" + urlencode({"q": f"site:redbus.in {query}"})),
            ("AbhiBus search", "https://www.google.com/search?" + urlencode({"q": f"site:abhibus.com {query}"})),
        ],
    }
    return providers[kind]


def transport_page(kind):
    origin = request.args.get("origin", "").strip()[:100]
    destination = request.args.get("destination", "").strip()[:100]
    travel_date = request.args.get("date", "").strip()[:10]
    searches = (
        build_external_searches(kind, origin, destination, travel_date)
        if origin and destination else []
    )
    return render_template(
        "transport.html",
        page_title=kind.capitalize(),
        kind=kind.capitalize(),
        endpoint=request.endpoint,
        origin=origin,
        destination=destination,
        travel_date=travel_date,
        searches=searches,
    )


@app.get("/flights")
def flights():
    return transport_page("flights")


@app.get("/trains")
def trains():
    return transport_page("trains")


@app.get("/buses")
def buses():
    return transport_page("buses")


@app.get("/hotels")
def hotels():
    search_text = request.args.get("q", "").strip()
    hotel_list = df[df["Category"] == "Hotel"].copy()
    if search_text:
        mask = hotel_list["Place"].str.contains(search_text, case=False, na=False) | hotel_list[
            "Description"
        ].str.contains(search_text, case=False, na=False)
        hotel_list = hotel_list[mask]
    return render_template(
        "hotels.html", page_title="Hotels", hotels=hotel_list, search_text=search_text,
    )


@app.route("/budget", methods=["GET", "POST"])
def budget_planner():
    values = {
        "days": 3,
        "travelers": 1,
        "budget": 10000,
        "hotel_per_night": DEFAULT_HOTEL_ESTIMATE,
        "meals_per_person_day": DEFAULT_MEALS_ESTIMATE,
        "activity_per_stop": DEFAULT_ACTIVITY_ESTIMATE,
    }
    plan = None
    error = None
    selected_interests = request.form.getlist("interests") if request.method == "POST" else default_interests
    if request.method == "POST":
        try:
            values["days"] = min(14, max(1, int(request.form.get("days", values["days"]))))
            values["travelers"] = min(20, max(1, int(request.form.get("travelers", values["travelers"]))))
            values["budget"] = max(0, int(request.form.get("budget", values["budget"])))
            values["hotel_per_night"] = max(0, int(request.form.get("hotel_per_night", DEFAULT_HOTEL_ESTIMATE)))
            values["meals_per_person_day"] = max(0, int(request.form.get("meals_per_person_day", DEFAULT_MEALS_ESTIMATE)))
            values["activity_per_stop"] = max(0, int(request.form.get("activity_per_stop", DEFAULT_ACTIVITY_ESTIMATE)))
        except ValueError:
            error = "Enter whole numbers for trip length, travelers, budget, and cost assumptions."
        if error is None:
            hotels_df = df[df["Category"] == "Hotel"]
            stops = df[~df["Category"].isin(SERVICE_CATEGORIES)]
            if selected_interests:
                preferred = stops[stops["Category"].isin(selected_interests)]
                other = stops[~stops["Category"].isin(selected_interests)]
                stops = pd.concat([preferred, other]).drop_duplicates("Place")
            stops = stops.head(MAX_ITINERARY_STOPS).copy()
            stops["Personalized_Score"] = [
                1.0 if category in selected_interests else 0.5
                for category in stops["Category"]
            ]
            stops["Along_Route_km"] = np.arange(len(stops), dtype=float)
            recommendations = pd.concat([hotels_df, stops], ignore_index=True)
            recommendations["Personalized_Score"] = recommendations["Personalized_Score"].fillna(0.5)
            itinerary = stops.copy()
            plan = build_budget_plan(
                recommendations, itinerary, values["days"], values["travelers"],
                values["budget"], values["hotel_per_night"],
                values["meals_per_person_day"], values["activity_per_stop"],
            )
    return render_template(
        "budget.html", page_title="Budget Planner", values=values, plan=plan,
        error=error, categories=categories, selected_interests=selected_interests,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("my_trips"))
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().casefold()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if not name or len(name) > 100 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            error = "Enter your name and a valid email address."
        elif len(password) < 8:
            error = "Choose a password with at least 8 characters."
        elif password != confirmation:
            error = "The passwords do not match."
        elif db.session.scalar(select(User.id).where(User.email == email)) is not None:
            error = "An account with that email already exists."
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                error = "An account with that email already exists."
            else:
                login_user(user)
                flash("Your account is ready. Plan a trip and save it to My Trips.", "success")
                return redirect(url_for("my_trips"))
    return render_template("auth.html", page_title="Register", mode="register", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("my_trips"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().casefold()
        password = request.form.get("password", "")
        user = db.session.scalar(select(User).where(User.email == email))
        if user is None or not user.check_password(password):
            error = "Email or password is incorrect."
        else:
            login_user(user)
            return redirect(url_for("my_trips"))
    return render_template("auth.html", page_title="Login", mode="login", error=error)


@app.post("/logout")
@login_required
def logout():
    logout_user()
    session.pop("latest_plan", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.get("/my-trips")
@login_required
def my_trips():
    trips = db.session.scalars(
        select(Trip).where(Trip.user_id == current_user.id).order_by(Trip.created_at.desc())
    ).all()
    return render_template("my_trips.html", page_title="My Trips", trips=trips)


@app.post("/trips/save")
@login_required
def save_trip():
    plan = session.get("latest_plan")
    if not plan:
        flash("Plan a trip first, then save it here.", "warning")
        return redirect(url_for("planner"))
    trip = Trip(
        user_id=current_user.id,
        title=f"{plan['start_name']} to {plan['dest_name']}",
        start_name=plan["start_name"],
        dest_name=plan["dest_name"],
        days=plan["days"],
        budget=plan["budget"],
        itinerary=plan["itinerary"],
    )
    db.session.add(trip)
    db.session.commit()
    flash("Trip saved to My Trips.", "success")
    return redirect(url_for("my_trips"))


@app.post("/trips/<int:trip_id>/delete")
@login_required
def delete_trip(trip_id):
    trip = db.session.scalar(
        select(Trip).where(Trip.id == trip_id, Trip.user_id == current_user.id)
    )
    if trip is None:
        abort(404)
    db.session.delete(trip)
    db.session.commit()
    flash("Trip deleted.", "info")
    return redirect(url_for("my_trips"))


@app.route("/planner", methods=["GET", "POST"])
def index():
    top_places = None
    itinerary = None
    route_summary = None
    start_coords = None
    dest_coords = None
    error_msg = None
    start_name = ""
    dest_name = ""
    days = 1
    travelers = 1
    budget = 10000
    hotel_per_night = DEFAULT_HOTEL_ESTIMATE
    meals_per_person_day = DEFAULT_MEALS_ESTIMATE
    activity_per_stop = DEFAULT_ACTIVITY_ESTIMATE
    budget_plan = None
    assistant_answer = None
    selected_interests = (
        default_interests
        if request.method == "GET"
        else [value for value in request.form.getlist("interests") if value in categories]
    )

    if request.method == "POST":
        session.pop("latest_plan", None)
        if request.form.get("action") == "assistant":
            assistant_answer = answer_travel_question(request.form.get("question", ""))
            return render_template(
                "index.html",
                top_places=None, itinerary=None, route_summary=None, budget_plan=None,
                error=None, locations=location_names, categories=categories,
                selected_interests=selected_interests, start_name=start_name,
                dest_name=dest_name, start_coords=None, dest_coords=None,
                assistant_answer=assistant_answer, question=request.form.get("question", ""),
                days=days, travelers=travelers, budget=budget,
                hotel_per_night=hotel_per_night,
                meals_per_person_day=meals_per_person_day,
                activity_per_stop=activity_per_stop,
            )

        try:
            days = min(14, max(1, int(request.form.get("days", 1))))
            travelers = min(20, max(1, int(request.form.get("travelers", 1))))
            budget = max(0, int(request.form.get("budget", 10000)))
            hotel_per_night = max(0, int(request.form.get("hotel_per_night", DEFAULT_HOTEL_ESTIMATE)))
            meals_per_person_day = max(0, int(request.form.get("meals_per_person_day", DEFAULT_MEALS_ESTIMATE)))
            activity_per_stop = max(0, int(request.form.get("activity_per_stop", DEFAULT_ACTIVITY_ESTIMATE)))
        except ValueError:
            error_msg = "Enter whole numbers for trip length, travelers, budget, and cost estimates."

        start_name = request.form.get("start_name", "").strip()
        dest_name = request.form.get("dest_name", "").strip()
        start_matches = df[df["Place"].str.casefold() == start_name.casefold()]
        dest_matches = df[df["Place"].str.casefold() == dest_name.casefold()]

        if error_msg:
            pass
        elif not start_name or not dest_name or start_matches.empty or dest_matches.empty:
            error_msg = "Choose a start and destination from the location suggestions."
        else:
            start_row = start_matches.iloc[0]
            dest_row = dest_matches.iloc[0]
            start_coords = (float(start_row["Latitude"]), float(start_row["Longitude"]))
            dest_coords = (float(dest_row["Latitude"]), float(dest_row["Longitude"]))
            try:
                route = get_road_route(start_coords, dest_coords)
            except RoutingError as exc:
                app.logger.warning("Road route request failed: %s", exc)
                error_msg = str(exc)
            else:
                route_places = measure_places_against_route(df, route["coordinates"])
                route_places = route_places[
                    route_places["Route_Distance_km"] <= ROUTE_CORRIDOR_KM
                ].copy()
                if route_places.empty:
                    error_msg = "No places from the current dataset were found within 15 km of this road route."
                    top_places = {}
                    itinerary = route_places
                else:
                    recommendations = build_recommendations(
                        route_places, start_coords, dest_coords, selected_interests
                    )
                    recommendations["Distance_km"] = recommendations["Route_Distance_km"]
                    tourist_places = recommendations[
                        ~recommendations["Category"].isin(SERVICE_CATEGORIES)
                    ]
                    top_places = {
                        "Tourist Places": tourist_places.sort_values(
                            "Personalized_Score", ascending=False
                        ),
                        "Hotels": recommendations[
                            recommendations["Category"] == "Hotel"
                        ].sort_values("Personalized_Score", ascending=False).head(5),
                        "Restaurants": recommendations[
                            recommendations["Category"] == "Restaurant"
                        ].sort_values("Personalized_Score", ascending=False).head(5),
                        "Petrol Stations": recommendations[
                            recommendations["Category"] == "Petrol Station"
                        ].sort_values("Personalized_Score", ascending=False).head(5),
                    }
                    itinerary = build_itinerary(recommendations, start_name, dest_name)
                    if days > 1 and not itinerary.empty:
                        itinerary["Day"] = [
                            min(days, (number - 1) // max(1, (len(itinerary) + days - 1) // days) + 1)
                            for number in range(1, len(itinerary) + 1)
                        ]
                    budget_plan = build_budget_plan(
                        recommendations, itinerary, days, travelers, budget,
                        hotel_per_night, meals_per_person_day, activity_per_stop,
                    )
                    itinerary = budget_plan.pop("itinerary")
                    if not itinerary.empty:
                        session["latest_plan"] = {
                            "start_name": start_name,
                            "dest_name": dest_name,
                            "days": days,
                            "budget": budget,
                            "itinerary": [
                                {
                                    "place": str(place.Place),
                                    "category": str(place.Category),
                                    "latitude": float(place.Latitude),
                                    "longitude": float(place.Longitude),
                                    "day": int(place.Day) if hasattr(place, "Day") else 1,
                                }
                                for place in itinerary.itertuples()
                            ],
                        }
                    if itinerary.empty:
                        error_msg = "The road route was found, but no tourist stops from the dataset are near it."

                route_summary = {
                    "distance_km": round(route["distance_km"], 1),
                    "duration_minutes": round(route["duration_minutes"]),
                }

    return render_template(
        "index.html",
        top_places=top_places,
        itinerary=itinerary,
        route_summary=route_summary,
        budget_plan=budget_plan,
        error=error_msg,
        locations=location_names,
        categories=categories,
        selected_interests=selected_interests,
        start_name=start_name,
        dest_name=dest_name,
        start_coords=start_coords,
        dest_coords=dest_coords,
        days=days,
        travelers=travelers,
        budget=budget,
        hotel_per_night=hotel_per_night,
        meals_per_person_day=meals_per_person_day,
        activity_per_stop=activity_per_stop,
        assistant_answer=assistant_answer,
        question="",
    )


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
