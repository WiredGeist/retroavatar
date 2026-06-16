import json

def read_sensor_data(shared_sensors: dict) -> str:
    """
    Decoupled utility to format the live shared sensors dictionary into JSON.
    """
    data = {
        "accelerometer": shared_sensors.get("a", [0, 0, 0]),
        "gyroscope": shared_sensors.get("g", [0, 0, 0]),
        "microphone_peak": shared_sensors.get("m", 0)
    }
    return json.dumps(data)
