import mpdaf
from mpdaf import obj
from mpdaf.obj import Cube
import numpy as np
from mpdaf.obj import iter_spe

cube=Cube("PathToCube")

def substract(cube):
    cont = cube.clone(data_init=np.empty, var_init=np.zeros)
    i=0
    shape=cube.shape
    step=int(shape[1]*shape[2]/10)
    print("Backgroundsubtraction:\n")
    #Anzeige in Prozent
    z=0
    for sp, co in zip(iter_spe(cube), iter_spe(cont)):
        i=i+1
        if sp.data.max() > 0:
            co[:] = sp.poly_spec(5)
        else:
            co[:] = None 
        
        #Anzeige in Prozent
        if (i%step)==0:
            z+=10
            print(z,"%")
        
    subtracted_cube = cube - cont
    subtracted_cube.write("YourPath")
    return
substract(cube)
