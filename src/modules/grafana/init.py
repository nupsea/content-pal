import os
import json
import requests

from dotenv import load_dotenv


load_dotenv()


GRAFANA_URL = "http://localhost:3000"

GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD")

PG_HOST = os.getenv("POSTGRES_HOST")
PG_DB = os.getenv("POSTGRES_DB")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_PORT = os.getenv("POSTGRES_PORT")


def create_api_key():
    auth = (GRAFANA_USER, GRAFANA_PASSWORD)
    headers = {"Content-Type": "application/json"}
    
    # Step 1: Create or get service account
    sa_payload = {
        "name": "ProgrammaticKey",
        "role": "Admin",
    }
    
    # Check if service account already exists
    sa_list_response = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/search", auth=auth)
    service_account_id = None
    
    if sa_list_response.status_code == 200:
        for sa in sa_list_response.json()["serviceAccounts"]:
            if sa["name"] == "ProgrammaticKey":
                service_account_id = sa["id"]
                print("Service account already exists")
                break
    
    # Create service account if it doesn't exist
    if service_account_id is None:
        sa_response = requests.post(
            f"{GRAFANA_URL}/api/serviceaccounts", auth=auth, headers=headers, json=sa_payload
        )
        if sa_response.status_code == 201:
            service_account_id = sa_response.json()["id"]
            print("Service account created successfully")
        elif sa_response.status_code == 400 and "already exists" in sa_response.text:
            # Service account exists but wasn't found in the list, try to get it again
            sa_list_response = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/search", auth=auth)
            if sa_list_response.status_code == 200:
                for sa in sa_list_response.json()["serviceAccounts"]:
                    if sa["name"] == "ProgrammaticKey":
                        service_account_id = sa["id"]
                        print("Found existing service account")
                        break
            if service_account_id is None:
                print("Service account exists but could not be found")
                return None
        else:
            print(f"Failed to create service account: {sa_response.text}")
            return None
    
    # Step 2: Check if token already exists, if so delete and recreate
    tokens_response = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/{service_account_id}/tokens", auth=auth)
    
    if tokens_response.status_code == 200:
        # Delete existing token if it exists
        for token in tokens_response.json():
            if token["name"] == "ProgrammaticToken":
                print("Deleting existing token...")
                delete_response = requests.delete(
                    f"{GRAFANA_URL}/api/serviceaccounts/{service_account_id}/tokens/{token['id']}", 
                    auth=auth
                )
                if delete_response.status_code == 200:
                    print("Existing token deleted")
                break
    
    # Create new token
    token_payload = {
        "name": "ProgrammaticToken"
    }
    
    token_response = requests.post(
        f"{GRAFANA_URL}/api/serviceaccounts/{service_account_id}/tokens",
        auth=auth, headers=headers, json=token_payload
    )
    
    if token_response.status_code == 200:
        print("Service account token created successfully")
        return token_response.json()["key"]
    else:
        print(f"Failed to create service account token: {token_response.text}")
        return None


def create_or_update_datasource(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    datasource_payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": f"postgres:{PG_PORT}",
        "access": "proxy",
        "user": PG_USER,
        "database": PG_DB,
        "basicAuth": False,
        "isDefault": True,
        "jsonData": {"sslmode": "disable", "postgresVersion": 1300},
        "secureJsonData": {"password": PG_PASSWORD},
    }

    print("Datasource payload:")
    print(json.dumps(datasource_payload, indent=2))

    # First, try to get the existing datasource
    response = requests.get(
        f"{GRAFANA_URL}/api/datasources/name/{datasource_payload['name']}",
        headers=headers,
    )

    if response.status_code == 200:
        # Datasource exists, let's update it
        existing_datasource = response.json()
        datasource_id = existing_datasource["id"]
        print(f"Updating existing datasource with id: {datasource_id}")
        response = requests.put(
            f"{GRAFANA_URL}/api/datasources/{datasource_id}",
            headers=headers,
            json=datasource_payload,
        )
    else:
        # Datasource doesn't exist, create a new one
        print("Creating new datasource")
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources", headers=headers, json=datasource_payload
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.text}")

    if response.status_code in [200, 201]:
        print("Datasource created or updated successfully")
        return response.json().get("datasource", {}).get("uid") or response.json().get(
            "uid"
        )
    else:
        print(f"Failed to create or update datasource: {response.text}")
        return None


def create_dashboard(api_key, datasource_uid):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    dashboard_file = "src/modules/grafana/dashboard.json"

    try:
        with open(dashboard_file, "r") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dashboard_file} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding {dashboard_file}: {str(e)}")
        return

    print("Dashboard JSON loaded successfully.")

    # Update datasource UID in the dashboard JSON
    panels_updated = 0
    for panel in dashboard_json.get("panels", []):
        if isinstance(panel.get("datasource"), dict):
            panel["datasource"]["uid"] = datasource_uid
            panels_updated += 1
        elif isinstance(panel.get("targets"), list):
            for target in panel["targets"]:
                if isinstance(target.get("datasource"), dict):
                    target["datasource"]["uid"] = datasource_uid
                    panels_updated += 1

    print(f"Updated datasource UID for {panels_updated} panels/targets.")

    # Remove keys that shouldn't be included when creating a new dashboard
    dashboard_json.pop("id", None)
    dashboard_json.pop("uid", None)
    dashboard_json.pop("version", None)

    # Prepare the payload
    dashboard_payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Updated by Python script",
    }

    print("Sending dashboard creation request...")

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=dashboard_payload
    )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code == 200:
        print("Dashboard created successfully")
        return response.json().get("uid")
    else:
        print(f"Failed to create dashboard: {response.text}")
        return None


def main():
    api_key = create_api_key()
    if not api_key:
        print("API key creation failed")
        return

    datasource_uid = create_or_update_datasource(api_key)
    if not datasource_uid:
        print("Datasource creation failed")
        return

    create_dashboard(api_key, datasource_uid)


if __name__ == "__main__":
    main()