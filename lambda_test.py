import requests

test_data = {
    "amount": [181.000000],
    "nameOrig": ["C840083671"],
    "oldbalanceOrg": [181.0],
    "nameDest": ["C38997010"],
    "newbalanceOrig": [0.0],
    "newbalanceDest": [0.0],
    "oldbalanceDest": [0.0],
    "step": [1],
    "type": ["TRANSFER"]
}

local_url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

stage_url = 'https://e6yc5urqh1.execute-api.eu-central-1.amazonaws.com/test/predict'

result = requests.post(local_url, json={ "transaction": test_data }).json()
print(result)
