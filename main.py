from crewai import Crew
from textwrap import dedent
from agents import TravelAgents
from tasks import TravelTasks

from dotenv import load_dotenv
load_dotenv()


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = TravelAgents()
        tasks = TravelTasks()

        # Define your custom agents and tasks here
        expert_travel_agent = agents.expert_travel_agent()
        city_selection_expert = agents.city_selection_expert()
        local_tour_guide = agents.local_tour_guide()

        # Custom tasks include agent name and variables as input
        plan_itinerary = tasks.plan_itinerary(
            expert_travel_agent,
            self.cities,
            self.date_range,
            self.interests
        )

        identify_city = tasks.identify_city(
            city_selection_expert,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_city_info = tasks.gather_city_info(
            local_tour_guide,
            self.cities,
            self.date_range,
            self.interests
        )

        # Define your custom crew here
        crew = Crew(
            agents=[expert_travel_agent,
                    city_selection_expert,
                    local_tour_guide
                    ],
            tasks=[
                plan_itinerary,
                identify_city,
                gather_city_info
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')
    origin = input(
        dedent("""
      From where will you be traveling from?
    """))
    cities = input(
        dedent("""
      What are the cities options you are interested in visiting?
    """))
    date_range = input(
        dedent("""
      What is the date range you are interested in traveling?
    """))
    interests = input(
        dedent("""
      What are some of your high level interests and hobbies?
    """))

    trip_crew = TripCrew(origin, cities, date_range, interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)
