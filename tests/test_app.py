import unittest
import uuid
from unittest.mock import patch

import app as travel_app
from extensions import db
from models import Trip, User


class TravelAppTests(unittest.TestCase):
    def setUp(self):
        travel_app.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        self.client = travel_app.app.test_client()
        self.email = f"test-{uuid.uuid4().hex}@example.com"
        with travel_app.app.app_context():
            db.create_all()

    def tearDown(self):
        with travel_app.app.app_context():
            user = db.session.scalar(db.select(User).where(User.email == self.email))
            if user is not None:
                db.session.delete(user)
                db.session.commit()

    def register(self):
        return self.client.post(
            "/register",
            data={
                "name": "Travel Test",
                "email": self.email,
                "password": "test-password-123",
                "confirm_password": "test-password-123",
            },
            follow_redirects=True,
        )

    def test_all_pages_and_external_searches_render(self):
        for path in (
            "/", "/explore", "/flights", "/trains", "/buses", "/hotels",
            "/planner", "/budget", "/login", "/register",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

        self.assertEqual(self.client.get("/my-trips").status_code, 302)
        response = self.client.get(
            "/flights?origin=Hyderabad&destination=Delhi&date=2026-12-01"
        )
        self.assertIn(b"Google Flights", response.data)
        self.assertIn(b"MakeMyTrip", response.data)
        self.assertEqual(self.client.get("/explore?q=fort").status_code, 200)
        self.assertEqual(self.client.get("/trains?origin=Hyderabad&destination=Warangal").status_code, 200)
        self.assertEqual(self.client.get("/buses?origin=Hyderabad&destination=Warangal").status_code, 200)

    def test_budget_planner_computes_an_estimate(self):
        response = self.client.post(
            "/budget",
            data={
                "days": "3",
                "travelers": "2",
                "budget": "10000",
                "hotel_per_night": "2500",
                "meals_per_person_day": "600",
                "activity_per_stop": "300",
                "interests": ["Historical"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated plan", response.data)
        self.assertIn(b"sample estimate", response.data.lower())

    def test_post_routes_reject_missing_csrf_token(self):
        travel_app.app.config["WTF_CSRF_ENABLED"] = True
        try:
            response = self.client.post("/budget", data={"days": "3"})
        finally:
            travel_app.app.config["WTF_CSRF_ENABLED"] = False
        self.assertEqual(response.status_code, 400)

    def test_user_can_register_plan_save_and_delete_trip(self):
        response = self.register()
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"My Trips", response.data)

        route = {
            "coordinates": [
                [78.4747, 17.3616],
                [78.60, 17.48],
                [78.75, 17.55],
                [78.8858, 17.5159],
                [78.9443, 17.6663],
            ],
            "distance_km": 75500,
            "duration_minutes": 5700,
        }
        with patch.object(travel_app, "get_road_route", return_value=route):
            planned = self.client.post(
                "/planner",
                data={
                    "action": "planner",
                    "start_name": "Charminar",
                    "dest_name": "Yadadri Temple",
                    "days": "3",
                    "travelers": "1",
                    "budget": "20000",
                    "hotel_per_night": "2000",
                    "meals_per_person_day": "600",
                    "activity_per_stop": "300",
                    "interests": ["Historical"],
                },
            )
        self.assertEqual(planned.status_code, 200)
        self.assertIn(b"Your suggested itinerary", planned.data)

        saved = self.client.post("/trips/save", follow_redirects=True)
        self.assertIn(b"Charminar to Yadadri Temple", saved.data)
        with travel_app.app.app_context():
            user = db.session.scalar(db.select(User).where(User.email == self.email))
            trip = db.session.scalar(db.select(Trip).where(Trip.user_id == user.id))
            trip_id = trip.id

        deleted = self.client.post(f"/trips/{trip_id}/delete", follow_redirects=True)
        self.assertEqual(deleted.status_code, 200)
        self.assertIn(b"No saved trips yet", deleted.data)
        self.client.post("/logout")
        login = self.client.post(
            "/login",
            data={"email": self.email, "password": "test-password-123"},
            follow_redirects=True,
        )
        self.assertEqual(login.status_code, 200)
        self.assertIn(b"My Trips", login.data)


if __name__ == "__main__":
    unittest.main()
