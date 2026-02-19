import requests
import json

api_key = "pbH2yvGNrNHuHG2GpIGwU6RH"
secret_key = "aVrFdSGaI7ZWzjtHiUvBOg5ruQipPWYt"

def main():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"

    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    main()
