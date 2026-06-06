import haversine as hs


class Location:
    # Class variable to keep track of the number of Location instances
    idx = 0

    def __init__(self, city: str, latitude: float, longitude: float):
        # Increment the class variable for each new instance
        Location.idx += 1
        # Assign a unique ID to the instance
        self.id = Location.idx
        self.city = city
        self.latitude = latitude
        self.longitude = longitude

    def get_distance_in_meters(self, distination: "Location"):
        source_location = (self.latitude, self.longitude)
        distination_location = (distination.latitude, distination.longitude)
        return hs.haversine(source_location, distination_location, hs.Unit.METERS)

    def get_distance_in_km(self, distination: "Location"):
        source_location = (self.latitude, self.longitude)
        distination_location = (distination.latitude, distination.longitude)
        return hs.haversine(source_location, distination_location)

    def save_as_dict(self) -> dict:
        # Return the instance variables as a dictionary
        return {"city": self.city, "latitude": self.latitude, "longitude": self.longitude}
