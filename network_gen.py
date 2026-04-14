import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_5g_data(hours=24):
    np.random.seed(42)
    data_points = hours * 60
    times = [datetime.now() - timedelta(minutes=i) for i in range(data_points)]

    user_count = (
        50
        + 30 * np.sin(np.linspace(0, 4 * np.pi, data_points))
        + np.random.normal(0, 5, data_points)
    )
    throughput = user_count * np.random.uniform(1, 5, data_points)
    rsrp = np.random.uniform(-110, -70, data_points)

    df = pd.DataFrame({
        'timestamp': times[::-1],
        'user_count': user_count.astype(int),
        'throughput_mbps': throughput,
        'rsrp_dbm': rsrp,
    })
    df.to_csv('network_data.csv', index=False)
    print("✅ Created network_data.csv")

if __name__ == "__main__":   # ← Fixed: was `name` (missing double underscores)
    generate_5g_data()