import io
import json
import os
import re
import time

import pandas as pd
import requests
from astropy.time import Time

PWF = os.path.expanduser("~/.atlas_fp")
ATLAS_BASEURL = "https://fallingstar-data.com/forcedphot"
ATLAS_OUTPUT_DIR = "atlas_lightcurves"
os.makedirs(ATLAS_OUTPUT_DIR, exist_ok=True)


def get_atlas_credentials():
    with open(PWF, "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()
        return username, password


def get_atlas_fp(ra, dec, mjd_min):
    username, password = get_atlas_credentials()

    resp = requests.post(
        url=f"{ATLAS_BASEURL}/api-token-auth/",
        data={"username": username, "password": password},
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}

    task_url = None
    while not task_url:
        with requests.Session() as s:
            resp = s.post(
                f"{ATLAS_BASEURL}/queue/",
                headers=headers,
                data={"ra": str(ra), "dec": str(dec), "mjd_min": mjd_min, "send_email": False},
            )
            if resp.status_code == 201:
                task_url = resp.json()["url"]
            elif resp.status_code == 429:
                message = resp.json()["detail"]
                print(f"{resp.status_code} {message}", flush=True)
                t_sec = re.findall(r"available in (\d+) seconds", message)
                t_min = re.findall(r"available in (\d+) minutes", message)
                waittime = int(t_sec[0]) if t_sec else (int(t_min[0]) * 60 if t_min else 10)
                print(f"Waiting {waittime} seconds", flush=True)
                time.sleep(waittime)
            else:
                print(f"ERROR {resp.status_code}", flush=True)
                print(resp.json(), flush=True)
                raise RuntimeError(f"ATLAS queue error {resp.status_code}")

    print("Task queued at", task_url, flush=True)
    result_url = None
    while not result_url:
        with requests.Session() as s:
            resp = s.get(task_url, headers=headers)
            if resp.status_code == 200:
                if resp.json()["finishtimestamp"]:
                    result_url = resp.json()["result_url"]
                    print(f"Task complete, results at {result_url}", flush=True)
                    break
                elif resp.json()["starttimestamp"]:
                    print(f"Task running (started {resp.json()['starttimestamp']})", flush=True)
                else:
                    print("Waiting for job to start...", flush=True)
                time.sleep(10)
            else:
                print(f"ERROR {resp.status_code}", flush=True)
                print(resp.json(), flush=True)
                raise RuntimeError(f"ATLAS poll error {resp.status_code}")

    with requests.Session() as s:
        textdata = s.get(result_url, headers=headers).text

    return pd.read_csv(io.StringIO(textdata.replace("###", "")), sep=r"\s+")


def write_atlas_lightcurve_to_file(name, atlas_fp_df):
    out_path = os.path.join(ATLAS_OUTPUT_DIR, f"{name}.json")
    records = atlas_fp_df.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {len(records)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    tns = pd.read_csv("tables/tns_public_objects_until_20250916.csv", header=0)
    row = tns[tns["name"] == "2020sgf"].iloc[0]
    ra, dec = row["ra"], row["declination"]
    mjd_min = Time(row["discoverydate"]).mjd

    print(f"Fetching ATLAS FP for 2020sgf: ra={ra}, dec={dec}, mjd_min={mjd_min}", flush=True)
    df = get_atlas_fp(ra, dec, mjd_min)
    write_atlas_lightcurve_to_file("2020sgf", df)
