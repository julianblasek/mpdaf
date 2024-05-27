import mpdaf
from mpdaf import obj
from mpdaf.obj import Cube
import numpy as np
from mpdaf.drs import PixTable
import matplotlib.pyplot as plt
import astropy.units as u
from mpdaf.obj import deg2sexa
from mpdaf.obj import Spectrum
from mpdaf.obj import iter_spe
import time
import random
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        delta_t = end_time - start_time
        
        # Convert delta_t to different time units
        if delta_t >= 60:
            minutes = int(delta_t // 60)
            seconds = delta_t % 60
            print(f"{func.__name__} ausgeführt in {minutes} Minuten und {seconds:.2f} Sekunden.")
        elif delta_t >= 1:
            print(f"{func.__name__} ausgeführt in {delta_t:.2f} Sekunden.")
        elif delta_t >= 1e-3:
            print(f"{func.__name__} ausgeführt in {delta_t * 1000:.2f} ms.")
        elif delta_t >= 1e-6:
            print(f"{func.__name__} ausgeführt in {delta_t * 1e6:.2f} µs.")
        else:
            print(f"{func.__name__} ausgeführt in {delta_t * 1e9:.2f} ns.")
            
        return result
    return wrapper
start_program = time.time()
@timed
def get_star_coord(cube,n):
    data = cube.sum(axis=0).data
    indices = np.argsort(data.ravel())[-n:]  # Sortiert und nimmt die letzten n Indizes
    yx_coords = np.unravel_index(indices, data.shape)
    return [(y, x) for y, x in zip(*yx_coords)]


#------------------- Constants and Globals ---------------------------------------------



path="PathToSavedPlot"
path2="PathToSavedTxt"
lambda_0 = 5007        #O_III
print("Es wurde Tor Nr.",elton," gewählt!")
c = 299792.458         #km/s
delta=18
flux_threshold=150
sn_threshold=2 
min_neighbors=3
discard_fraction=0.1
aperture_radius = 12  # in Pixeln
t_sn_seconds = 56 * 365.25 * 24 * 3600 #a=56
        
min=lambda_0-delta        #min
max=lambda_0+delta        #max
v_max=c*delta/lambda_0
v_max_expansion=5000

#path background subtracted path
cube_path="BackSubCube"

name = "HR_DEL"

fullcube=Cube(cube_path)
#read in background subtracted cube
cube=Cube(cube_path).select_lambda(min, max)
cont_cube=Cube(cont_path)
depth_3d=cube[:, 30, 30].data.shape[0]
#print(depth_3d)
#5 most bright spots
center_list=list(reversed(get_star_coord(fullcube,5)))
x_center = center_list[0][1]
y_center = center_list[0][0]
theta = 0.2  # Winkelauflösung pro Pixel in Bogensekunden
d = 897  # Distanz Beobachter - Stern in pc

#----------------------------------------------------------------------------------------------
@timed
def plot_sn_map(map):
    plt.figure(figsize=(10, 8))
    plt.imshow(np.mean(map, axis=0), origin='lower', cmap='viridis')
    plt.colorbar(label='Signal-to-Noise Ratio')
    plt.title('Signal-to-Noise Ratio Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    #plt.scatter(20, 60, color='red', s=70)
    plt.show()
    
    return

#----------------------------------- Mask Star -------------------------------------------------------------
@timed
def mask_star(cube, star_position, aperture_radius):
    masked_cube = cube.copy()  # Kopie des Cubes erstellen, um das Original nicht zu verändern
    y, x = np.ogrid[:cube.shape[1], :cube.shape[2]]  # Koordinatengitter erstellen

    # Entfernungsquadrat vom Sternzentrum berechnen
    distance_sq = (x - star_position[1])**2 + (y - star_position[0])**2

    # Maske erstellen, die True innerhalb der Apertur ist
    mask = distance_sq <= (aperture_radius**2)

    # Alle Spektren innerhalb der Apertur maskieren
    for i in range(cube.shape[0]):  # Über jede Schicht des Cubes iterieren
        masked_cube.data[i, mask] = np.nan  # NaN setzen, um die Sternregion zu maskieren

    return masked_cube
 
#----------------------------------- Pixel Criteria -------------------------------------------------------------
@timed
def extract_nova_shell(cube):
    path_sn_map = f"{path}sn_map.pdf"
    line_cube = cube.select_lambda(min, max)


    # Berechnen Sie S/N für jeden Spaxel im Subcube
    sn_map = abs(line_cube.data / np.sqrt(line_cube.var))
    
    #np.savetxt(f"{path}sn_map.txt", sn_map.mean(axis=0), fmt='%.6f', delimiter='\t', header='Signal-to-Noise Ratio')
    #plot_sn_map(sn_map)

    # Erste Auswahlkriterium: S/N-Schwelle
    signal_mask = sn_map > sn_threshold

    # Zweites Kriterium: Mindestanzahl benachbarter Spaxels
    neighbor_mask = ndimage.generic_filter(signal_mask.astype(int), np.sum, size=3, mode='constant', cval=0) >= min_neighbors

    # Kombinieren Sie beide Kriterien
    final_mask = signal_mask & neighbor_mask

    
    sorted_flux = np.sort(np.abs(line_cube.data[final_mask]))
    cutoff_index = int(discard_fraction * len(sorted_flux))
    flux_threshold = sorted_flux[cutoff_index]
    # Anwenden des Schwellenwerts auf die ursprünglichen Daten
    final_mask &= np.abs(line_cube.data) >= flux_threshold



    # Anwenden der endgültigen Maske
    filtered_cube = line_cube.copy()
    filtered_cube.data[~final_mask] = 0  # Null setzen, wo die Maske nicht erfüllt ist

    return filtered_cube

#----------------------------------- Geschwindigkeit bestimmen -------------------------------------------------------------
@timed
def calculate_velocity_fields(cube, unit=u.angstrom):
    #filename_doppler = f"{path}doppler_shifts.txt"
    #filename_expansion = f"{path}expansionsgeschwindigkeit.txt"

    # Initialization of 3D arrays for velocity fields
    doppler_shifts_velocity = np.zeros((cube.shape[1], cube.shape[2], depth_3d))
    Rz = np.zeros((cube.shape[1], cube.shape[2], depth_3d))
    flux_grid = np.zeros((cube.shape[1], cube.shape[2], depth_3d))
    
    
    
    # Iterate over each pixel
    for i in range(cube.shape[1]):  # X-axis
        for j in range(cube.shape[2]):  # Y-axis
            spec = cube[:, i, j]
            wavelengths = spec.wave.coord()  # Get wavelengths
            doppler_shifts = c * ((wavelengths - lambda_0) / lambda_0)
            # Assign calculated values to the arrays
            doppler_shifts_velocity[i, j, :] = doppler_shifts
            Rz[i, j, :]=doppler_shifts*t_sn_seconds
            flux_grid[i, j, :] = spec.data


    # Further processing and saving of velocity fields can be done here
    return doppler_shifts_velocity, flux_grid, Rz

@timed
def get_distances(cube,Rz,flux):
    expansionsgeschwindigkeiten = np.zeros((cube.shape[1], cube.shape[2], depth_3d))
    # Berechne Rx und Ry

    with open(path2, 'w') as file:
        file.write('x_pixel,y_pixel,R_x(pc),R_y(pc),R_z(pc),v_exp(km/s,)Flux()\n')
        flux_max=0
        for y in range(cube.shape[1]):
            for x in range(cube.shape[2]):
                for z in range(depth_3d):
                    rel_x = x - x_center
                    rel_y = y - y_center
                    Rx = (rel_x * theta / 206265) * d  # Umrechnung in pc
                    Ry = (rel_y * theta / 206265) * d  # Umrechnung in pc
                    R_z=(Rz[y ,x, z] / 3.086e13)
                    R_ges=np.sqrt(Rx**2+Ry**2+R_z**2)*3.086e13
                    R_z=random.uniform(-1, 1)*0.0045+R_z
                        
                    v_exp=R_ges/t_sn_seconds
                    expansionsgeschwindigkeiten[y ,x, z]=v_exp
                    flux_actual=flux[y ,x, z]
                    if(flux_actual>flux_max):
                        flux_max=flux_actual
                    if(flux_actual>flux_threshold):
                        file.write(f'{x},{y},{Rx:.6f},{Ry:.6f},{R_z:.6f},{v_exp:.6f},{flux_actual:.6f}\n')
                    
    return flux_max

 
def main():
    filtered_cube=extract_nova_shell(cube)
    masked_cube=mask_star(filtered_cube,(y_center,x_center),aperture_radius)
    doppler_velocity, flux_all, Rz = calculate_velocity_fields(masked_cube)
    flux_max=get_distances(masked_cube,Rz,flux_all)
    print("maximaler Flux: ",flux_max)
    end_program = time.time()
    print(f"Runtime: {end_program - start_program} Sekunden.")
    return 
#main()
