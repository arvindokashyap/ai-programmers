"""
Trip Planning System with Human-in-the-Loop

This script demonstrates a simple HITL system for planning a trip with three steps:
1. Destination selection with LLM suggestion
2. Flight selection with LLM suggestion
3. Itinerary generation with LLM suggestion

Each step includes human verification and modification options.

HITL checkpoints implemented in plan_trip:
- Destination: confirm the suggestion or type your own.
- Flight: confirm, regenerate, or enter the flight details manually. Flights cost
  money and are hard to undo, so the human always has the final say here.
- Itinerary: confirm, or give feedback that is fed back into the LLM for a new
  draft (up to a limit), or write the days yourself.
- Final review: the whole plan is shown and must be approved before it is saved.
  Saving to disk is the only side effect in this script, so it gets its own gate.
"""

import json
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI
import os
from pydantic import BaseModel, Field

# Rocket's TLS-inspecting firewall needs the Windows cert store
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

MODEL = "gpt-4o-mini-2024-07-18"
MAX_REGENERATIONS = 3
TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

@dataclass
class TripPlan:
    """Represents a complete trip plan."""
    destination: str
    departure_city: str
    flight_info: "FlightInfo"
    itinerary: "Itinerary"

class FlightInfo(BaseModel):
    """Model for flight information."""
    flight_number: str = Field(description="Flight number")
    departure_time: str = Field(description="Departure time in HH:MM format")
    arrival_time: str = Field(description="Arrival time in HH:MM format")
    airline: str = Field(description="Name of the airline")

class Itinerary(BaseModel):
    """Model for daily itinerary."""
    daily_activities: List[List[str]] = Field(description="List of activities for each day of the trip")

class TripPlanner:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.trip_plan = None

    # ------------------------------------------------------------------ LLM calls

    def suggest_destination(self) -> str:
        """Use LLM to suggest a travel destination."""
        prompt = """Suggest an interesting travel destination.
        Consider factors like weather, tourist attractions, and cultural experiences.
        Return only the destination name, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting destination suggestion: {e}")
            return "Paris"  # Fallback destination

    def suggest_flight(self, departure_city: str, destination: str) -> FlightInfo:
        """Use LLM to suggest a fictional flight."""
        prompt = f"""Generate a fictional flight from {departure_city} to {destination}.
        Include departure time, arrival time, and flight number."""

        try:
            response = self.client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                text_format=FlightInfo
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error getting flight suggestion: {e}")
            # Return the same type as the success path so callers never have to branch
            return FlightInfo(
                flight_number="AA123",
                departure_time="10:00",
                arrival_time="12:00",
                airline="Example Airlines",
            )

    def generate_itinerary(self, destination: str, feedback: Optional[str] = None) -> Itinerary:
        """Use LLM to generate a brief itinerary, optionally steered by human feedback."""
        prompt = f"""Generate a brief 3-day itinerary for {destination}.
        Include 2-3 activities per day."""
        if feedback:
            prompt += f"\n\nThe traveler reviewed a previous draft and asked for these changes: {feedback}"

        try:
            response = self.client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                text_format=Itinerary
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error generating itinerary: {e}")
            return Itinerary(daily_activities=[
                ["Morning activity", "Afternoon activity"],
                ["Morning activity", "Afternoon activity"],
                ["Morning activity", "Afternoon activity"],
            ])

    # ------------------------------------------------------------- human input

    def get_human_confirmation(self, prompt: str) -> bool:
        """Get human confirmation for a suggestion."""
        while True:
            choice = input(f"{prompt} (y/n): ").lower().strip()
            if choice in ['y', 'n']:
                return choice == 'y'
            print("Please enter 'y' or 'n'")

    def get_human_choice(self, prompt: str, options: Dict[str, str]) -> str:
        """Ask the human to pick one of several lettered options."""
        menu = ", ".join(f"[{key}] {label}" for key, label in options.items())
        while True:
            choice = input(f"{prompt} {menu}: ").lower().strip()
            if choice in options:
                return choice
            print(f"Please enter one of: {', '.join(options)}")

    def get_text_input(self, prompt: str, validator=None, error: str = "Please enter a value.") -> str:
        """Ask for free text and keep asking until it is non-empty and passes the validator."""
        while True:
            value = input(prompt).strip()
            if value and (validator is None or validator(value)):
                return value
            print(error)

    def get_manual_flight(self) -> FlightInfo:
        """Let the human type in a flight they already found."""
        print("\nEnter your flight details:")
        is_time = lambda value: bool(TIME_PATTERN.match(value))
        return FlightInfo(
            airline=self.get_text_input("  Airline: "),
            flight_number=self.get_text_input("  Flight number: "),
            departure_time=self.get_text_input("  Departure time (HH:MM, 24h): ", is_time, "  Use HH:MM, for example 08:30 or 17:45."),
            arrival_time=self.get_text_input("  Arrival time (HH:MM, 24h): ", is_time, "  Use HH:MM, for example 08:30 or 17:45."),
        )

    def get_manual_itinerary(self, days: int = 3) -> Itinerary:
        """Let the human type in their own activities, one line per activity."""
        print("\nEnter your own itinerary. Type activities one per line; press Enter on an empty line to finish a day.")
        daily_activities: List[List[str]] = []
        for day in range(1, days + 1):
            print(f"\nDAY {day}:")
            activities: List[str] = []
            while True:
                activity = input("  - ").strip()
                if not activity:
                    if activities:
                        break
                    print("  Add at least one activity for this day.")
                    continue
                activities.append(activity)
            daily_activities.append(activities)
        return Itinerary(daily_activities=daily_activities)

    # ----------------------------------------------------------------- display

    def print_flight(self, flight: FlightInfo) -> None:
        print(f"Airline: {flight.airline}")
        print(f"Flight: {flight.flight_number}")
        print(f"Departure: {flight.departure_time}")
        print(f"Arrival: {flight.arrival_time}")

    def print_itinerary(self, itinerary: Itinerary) -> None:
        for day, activities in enumerate(itinerary.daily_activities, 1):
            print(f"\nDAY {day}:")
            for activity in activities:
                print(f"- {activity}")

    def print_plan(self, plan: TripPlan) -> None:
        print(f"Destination: {plan.destination}")
        print(f"Departure from: {plan.departure_city}")
        print("\nFlight Information:")
        self.print_flight(plan.flight_info)
        print("\nItinerary:")
        self.print_itinerary(plan.itinerary)

    # ------------------------------------------------------------------- flow

    def plan_trip(self) -> None:
        """Run the complete trip planning process."""
        print("\n=== Welcome to the Trip Planner! ===\n")

        # Step 1: Destination Selection
        # Checkpoint: a destination the traveler does not want makes every later
        # step useless, so confirm it before spending any more LLM calls.
        suggested_destination = self.suggest_destination()
        print(f"\nSuggested destination: {suggested_destination}")

        if self.get_human_confirmation("Would you like to use this destination?"):
            destination = suggested_destination
        else:
            destination = self.get_text_input("Enter your preferred destination: ", error="Destination cannot be empty.")

        # Step 2: Flight Selection
        # Checkpoint: flights are a real-money, hard-to-reverse decision. The
        # traveler can accept, ask for another suggestion, or enter their own.
        departure_city = self.get_text_input("\nEnter your departure city: ", error="Departure city cannot be empty.")
        flight = self.suggest_flight(departure_city, destination)
        regenerations = 0
        while True:
            print("\nSuggested flight:")
            self.print_flight(flight)

            if self.get_human_confirmation("Would you like to use this flight?"):
                break

            options = {"m": "enter my own flight"}
            if regenerations < MAX_REGENERATIONS:
                options = {"r": "suggest another flight", **options}
            choice = self.get_human_choice("What would you like to do?", options)
            if choice == "r":
                regenerations += 1
                flight = self.suggest_flight(departure_city, destination)
            else:
                flight = self.get_manual_flight()
                break

        # Step 3: Itinerary Generation
        # Checkpoint: itineraries are subjective, so instead of a plain accept /
        # reject the traveler can say what to change and the LLM redrafts.
        itinerary = self.generate_itinerary(destination)
        regenerations = 0
        while True:
            print("\nSuggested itinerary:")
            self.print_itinerary(itinerary)

            if self.get_human_confirmation("\nWould you like to use this itinerary?"):
                break

            options = {"m": "write my own itinerary"}
            if regenerations < MAX_REGENERATIONS:
                options = {"f": "give feedback and regenerate", **options}
            choice = self.get_human_choice("What would you like to do?", options)
            if choice == "f":
                regenerations += 1
                feedback = self.get_text_input("What should change? ", error="Please describe what you want changed.")
                itinerary = self.generate_itinerary(destination, feedback=feedback)
            else:
                itinerary = self.get_manual_itinerary()
                break

        # Build the complete trip plan
        self.trip_plan = TripPlan(
            destination=destination,
            departure_city=departure_city,
            flight_info=flight,
            itinerary=itinerary
        )

        # Step 4: Final review
        # Checkpoint: the traveler sees everything together once before anything
        # is persisted. Saving is the only side effect, so it is gated separately.
        print("\n=== Your Trip Plan ===")
        self.print_plan(self.trip_plan)

        if not self.get_human_confirmation("\nDoes this plan look right?"):
            print("\nNo problem. Nothing was saved; run the planner again to start over.")
            self.trip_plan = None
            return

        if self.get_human_confirmation("Save this plan to trip_plan.json?"):
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trip_plan.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "destination": self.trip_plan.destination,
                    "departure_city": self.trip_plan.departure_city,
                    "flight_info": self.trip_plan.flight_info.model_dump(),
                    "itinerary": self.trip_plan.itinerary.model_dump(),
                }, f, indent=2)
            print(f"Saved to {output_path}")
        else:
            print("Plan confirmed but not saved.")

def main():
    planner = TripPlanner()
    try:
        planner.plan_trip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nPlanning cancelled. Nothing was saved.")

if __name__ == "__main__":
    main()
