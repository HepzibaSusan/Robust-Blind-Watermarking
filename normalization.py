import numpy as np

# Original watermark bits
originals = [
    [1,0,1,1,0],
    [1,0,0,1,1],
    [1,0,0,0,1]
]

# Extracted watermarks under attacks
jpeg = [
    [1,0,1,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0]
]

noise = [
    [1,0,1,1,0],
    [1,0,0,1,1],
    [1,0,0,0,1]
]

rotation = [
    [1,1,0,0,0],
    [1,0,0,1,1],
    [1,0,0,1,0]
]

scaling = [
    [1,0,0,0,1],
    [0,0,1,0,0],
    [1,0,0,0,0]
]

# Function to map 0 -> -1, 1 -> 1
def to_pm1(bits):
    return [1 if b==1 else -1 for b in bits]

def compute_nc(orig, extracted):
    orig_pm1 = to_pm1(orig)
    ext_pm1 = to_pm1(extracted)
    orig_pm1 = np.array(orig_pm1)
    ext_pm1 = np.array(ext_pm1)
    nc = np.sum(orig_pm1 * ext_pm1) / np.sum(orig_pm1**2)
    return nc

# Compute NC for each attack and watermark
attacks = {'JPEG': jpeg, 'Noise': noise, 'Rotation': rotation, 'Scaling': scaling}

for attack_name, attack_data in attacks.items():
    nc_vals = [compute_nc(orig, ext) for orig, ext in zip(originals, attack_data)]
    print(f"{attack_name} NC values:", nc_vals)
    print(f"Average NC for {attack_name}: {np.mean(nc_vals):.3f}")
