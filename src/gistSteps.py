
# Step 1: User Selects Earthquake
# INPUT:
# {
#   eventId: '',
#   earthquakeDetails: {
#   ...
#   } 
# }


def step1(input):

    api_url = http://scdb.beg.utexas.edu/fdsnws/event/1/builder
    if input.eventId != '':
        response = requests.get(api_url, verify=False)

    earthquake_info = {
        'Event ID': event_id,
        'Latitude': float(latitude),
        'Longitude': float(longitude),
        'Origin Date': origin_date,
        'Origin Time': datetime.datetime.strptime(origin_time, '%H:%M:%S.%f').strftime('%H:%M:%S'),
        'Local Magnitude': round(float(magnitude), 2)
    }
    return earthquake_info




def step2(input):

