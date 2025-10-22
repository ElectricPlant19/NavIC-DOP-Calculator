
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO

#Navic Sats
NAVIK_SATS = {
    "IRNSS-1B": 39635,
    "IRNSS-1C": 40269,
    "IRNSS-1D": 40547,
    "IRNSS-1E": 41241,
    "IRNSS-1F": 41384,
    "IRNSS-1I": 43286,
    "NVS-01": 56759
}

# Extreme points of India
INDIA_EXTREME_POINTS = {
    "Northernmost (Siachen Glacier)": (35.5, 77.0),
    "Southernmost (Indira Point)": (6.75, 93.85),
    "Easternmost (Kibithu)": (28.0, 97.0),
    "Westernmost (Guhar Moti)": (23.7, 68.1),
    "Capital (Delhi)": (28.7, 77.1)
}


class SpaceTrackClient:
    """Client for Space-Track.org API"""
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "https://www.space-track.org"
        self.session = None
    
    def login(self):
        """Login to Space-Track"""
        self.session = requests.Session()
        login_url = f"{self.base_url}/ajaxauth/login"
        data = {
            'identity': self.username,
            'password': self.password
        }
        try:
            response = self.session.post(login_url, data=data)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            return False
    
    def get_tle(self, norad_id):
        """Get TLE data for a specific NORAD ID"""
        if not self.session:
            if not self.login():
                return None
        
        query_url = f"{self.base_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/TLE_LINE1 ASC/format/3le"
        
        try:
            response = self.session.get(query_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Error fetching TLE for {norad_id}: {str(e)}")
            return None
    
    def get_multiple_tles(self, norad_ids):
        """Get TLE data for multiple NORAD IDs"""
        if not self.session:
            if not self.login():
                return {}
        
        # Join NORAD IDs with comma
        ids_str = ','.join(map(str, norad_ids))
        query_url = f"{self.base_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{ids_str}/orderby/NORAD_CAT_ID,ORDINAL/format/3le"
        
        try:
            response = self.session.get(query_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Error fetching TLEs: {str(e)}")
            return ""
        

def parse_tle_data(tle_text, sat_dict):
    """Parse TLE text and create satellite objects"""
    ts = load.timescale()
    satellites = {}
    
    lines = tle_text.strip().split('\n')
    
    # Process TLE in groups of 3 (name, line1, line2)
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        
        # Extract NORAD ID from line 1
        try:
            norad_id = int(line1[2:7])
            
            # Find satellite name from our dictionary
            sat_name = None
            for s_name, s_id in sat_dict.items():
                if s_id == norad_id:
                    sat_name = s_name
                    break
            
            if sat_name:
                satellite = EarthSatellite(line1, line2, sat_name, ts)
                satellites[sat_name] = satellite
        except (ValueError, IndexError) as e:
            st.warning(f"Error parsing TLE: {str(e)}")
            continue
    
    return satellites


def calculate_satellite_position(satellite, time, observer_location):
    """Calculate satellite position relative to observer"""
    try:
        difference = satellite - observer_location
        topocentric = difference.at(time)
        alt, az, distance = topocentric.altaz()
        
        return {
            'altitude': alt.degrees,
            'azimuth': az.degrees,
            'distance': distance.km,
            'elevation': alt.degrees
        }
    except Exception as e:
        st.warning(f"Error calculating position: {str(e)}")
        return None
    

def calculate_design_matrix(satellite_positions, observer_lat, observer_lon):
    """
    Calculate the geometry matrix (design matrix) for DOP calculation
    H matrix where each row represents the unit vector from receiver to satellite
    """
    H = []
    
    for pos in satellite_positions:
        if pos is None:
            continue
            
        # Only include satellites above horizon (elevation > 5 degrees for better geometry)
        if pos['elevation'] > 5:
            # Convert azimuth and elevation to direction cosines
            az_rad = np.radians(pos['azimuth'])
            el_rad = np.radians(pos['elevation'])
            
            # Direction cosines (East, North, Up)
            dx = np.cos(el_rad) * np.sin(az_rad)
            dy = np.cos(el_rad) * np.cos(az_rad)
            dz = np.sin(el_rad)
            
            # Add row: [dx, dy, dz, 1] - the 1 is for clock bias
            H.append([dx, dy, dz, 1])
    
    return np.array(H) if H else np.array([]).reshape(0, 4)

def calculate_dop_values(H):
    """
    Calculate various DOP values from the design matrix
    DOP = sqrt(trace(Q)) where Q = (H^T * H)^-1
    """
    if len(H) < 4:
        return None  # Need at least 4 satellites
    
    try:
        # Calculate (H^T * H)^-1
        HTH = np.dot(H.T, H)
        
        # Check if matrix is singular
        if np.linalg.det(HTH) == 0:
            return None
            
        Q = np.linalg.inv(HTH)
        
        # Extract DOP values
        dop = {
            'GDOP': float(np.sqrt(np.trace(Q))),  # Geometric DOP
            'PDOP': float(np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])),  # Position DOP
            'HDOP': float(np.sqrt(Q[0,0] + Q[1,1])),  # Horizontal DOP
            'VDOP': float(np.sqrt(Q[2,2])),  # Vertical DOP
            'TDOP': float(np.sqrt(Q[3,3])),  # Time DOP
        }
        
        return dop
    except np.linalg.LinAlgError:
        return None  # Singular matrix
    except Exception as e:
        st.warning(f"Error calculating DOP: {str(e)}")
        return None
    

def calculate_dop_for_location(satellites_dict, lat, lon, time):
    """Calculate DOP for a specific location and time"""
    ts = load.timescale()
    t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
    
    # Create observer location
    observer = wgs84.latlon(lat, lon)
    
    # Calculate positions for all satellites
    satellite_positions = []
    visible_sats = []
    
    for sat_name, sat_obj in satellites_dict.items():
        pos = calculate_satellite_position(sat_obj, t, observer)
        if pos:
            satellite_positions.append(pos)
            if pos['elevation'] > 5:  # 5 degree elevation mask
                visible_sats.append(sat_name)
    
    # Calculate design matrix
    H = calculate_design_matrix(satellite_positions, lat, lon)
    
    # Calculate DOP
    dop = calculate_dop_values(H)
    
    return dop, visible_sats, satellite_positions



def main():
    """Main function to test NavIC DOP calculations"""
    
    print("=" * 80)
    print("NavIC Satellite DOP Calculator - Command Line Test")
    print("=" * 80)
    print()
    
    # Configuration
    print("Enter Space-Track.org credentials:")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    print()
    
    # Initialize Space-Track client
    print("Connecting to Space-Track.org...")
    client = SpaceTrackClient(username, password)
    
    if not client.login():
        print("Failed to login to Space-Track.org")
        return
    
    print("✓ Successfully connected")
    print()
    
    # Fetch TLE data
    print("Fetching TLE data for NavIC satellites...")
    norad_ids = list(NAVIK_SATS.values())
    tle_data = client.get_multiple_tles(norad_ids)
    
    if not tle_data:
        print("Failed to fetch TLE data")
        return
    
    print(f"✓ Retrieved TLE data for {len(NAVIK_SATS)} satellites")
    print()
    
    # Parse TLE data
    print("Parsing TLE data...")
    satellites = parse_tle_data(tle_data, NAVIK_SATS)
    print(f"✓ Successfully parsed {len(satellites)} satellites")
    print()
    
    # Display satellite list
    print("NavIC Satellites:")
    print("-" * 40)
    for sat_name in satellites.keys():
        norad_id = NAVIK_SATS[sat_name]
        print(f"  • {sat_name} (NORAD ID: {norad_id})")
    print()
    
    # Calculate DOP for extreme points of India
    print("=" * 80)
    print("Calculating DOP for Extreme Points of India")
    print("=" * 80)
    print()
    
    current_time = datetime.utcnow()
    print(f"Calculation Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    for location_name, (lat, lon) in INDIA_EXTREME_POINTS.items():
        print(f"\n{location_name}")
        print(f"  Location: {lat}°N, {lon}°E")
        print("-" * 60)
        
        dop, visible_sats, sat_positions = calculate_dop_for_location(
            satellites, lat, lon, current_time
        )
        
        print(f"  Visible Satellites (>5° elevation): {len(visible_sats)}")
        
        if visible_sats:
            print("    Satellites:")
            for sat in visible_sats:
                # Find position for this satellite
                sat_idx = list(satellites.keys()).index(sat)
                if sat_idx < len(sat_positions):
                    pos = sat_positions[sat_idx]
                    print(f"      - {sat}: El={pos['elevation']:.1f}°, Az={pos['azimuth']:.1f}°, Dist={pos['distance']:.0f}km")
        
        print()
        
        if dop:
            print("  DOP Values:")
            print(f"    GDOP (Geometric):  {dop['GDOP']:.2f}")
            print(f"    PDOP (Position):   {dop['PDOP']:.2f}")
            print(f"    HDOP (Horizontal): {dop['HDOP']:.2f}")
            print(f"    VDOP (Vertical):   {dop['VDOP']:.2f}")
            print(f"    TDOP (Time):       {dop['TDOP']:.2f}")
            
            # DOP quality assessment
            gdop = dop['GDOP']
            if gdop < 2:
                quality = "Excellent"
            elif gdop < 4:
                quality = "Good"
            elif gdop < 6:
                quality = "Moderate"
            elif gdop < 8:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"    Quality: {quality}")
            
            results.append({
                'location': location_name,
                'lat': lat,
                'lon': lon,
                'visible_sats': len(visible_sats),
                'gdop': dop['GDOP'],
                'quality': quality
            })
        else:
            print("  DOP Values: Cannot calculate (insufficient satellites)")
            results.append({
                'location': location_name,
                'lat': lat,
                'lon': lon,
                'visible_sats': len(visible_sats),
                'gdop': None,
                'quality': 'N/A'
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Location':<35} {'Lat':>8} {'Lon':>8} {'Sats':>5} {'GDOP':>8} {'Quality':>10}")
    print("-" * 80)
    
    for result in results:
        gdop_str = f"{result['gdop']:.2f}" if result['gdop'] else "N/A"
        print(f"{result['location']:<35} {result['lat']:>8.2f} {result['lon']:>8.2f} "
              f"{result['visible_sats']:>5} {gdop_str:>8} {result['quality']:>10}")
    
    print()
    print("=" * 80)
    print("DOP Quality Guide:")
    print("  Excellent: GDOP < 2  |  Good: 2-4  |  Moderate: 4-6  |  Fair: 6-8  |  Poor: >8")
    print("=" * 80)
    print()


if __name__ == "__main__":

    main()
