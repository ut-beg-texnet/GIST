
# Step 1: User Selects Earthquake
# INPUT:
# {
#   eventId: '',
#   earthquakeDetails: {
#   ...
#   } 
# }

api_url = http://scdb.beg.utexas.edu/fdsnws/event/1/builder

def step1(input):

    def get_earthquake_info_from_csv(csv_string):
    # Parse the CSV string and extract earthquake information
    reader = csv.reader(csv_string.splitlines())
    rows = list(reader)
    event_id, origin_datetime, latitude, longitude, _, magnitude, _ = rows[0]
    origin_datetime = origin_datetime.replace('Z', '')
    origin_date, origin_time = origin_datetime.split('T')

    earthquake_info = {
        'Event ID': event_id,
        'Latitude': float(latitude),
        'Longitude': float(longitude),
        'Origin Date': origin_date,
        'Origin Time': datetime.datetime.strptime(origin_time, '%H:%M:%S.%f').strftime('%H:%M:%S'),
        'Local Magnitude': round(float(magnitude), 2)
    }
    return earthquake_info

    if input.eventId != '':
        response = requests.get(api_url, verify=False)