# -*- coding: utf-8 -*-
"""
Recopilación de funciones que están usadas en el programa (y se han ido arreglando)
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from astropy.io import fits



# FUNCIONES PARA PLOTS

def show_data(data, permin=0.1, permax=99.9, cmapp = 'bone', cbarname = ' ', plot_name = ' '):
    plt.figure(figsize=(10, 7))
    plt.imshow(data, vmin=np.nanpercentile(data,permin), vmax=np.nanpercentile(data,permax)
               , origin='lower' , cmap = cmapp)
    cbar = plt.colorbar()
    cbar.set_label(cbarname)   
    plt.title(plot_name)
    plt.show()
    plt.close()


def show_data_log(data_nor, permin = 8, permax = 99.9, cmapp = 'viridis', cbarname = ' '):
    plt.figure(figsize=(10, 7))
    data = np.copy(data_nor)
    # CAMBIO DE NEGATIVOS POR 0
    aneg = np.where(data<0)
    data[aneg] = 0
    # OFFSET
    aoffset = np.where(data>0)
    sin_ceros = data[aoffset]
    offset = np.nanmin(sin_ceros, axis = None)    
    data = data+offset #(Para evitar los ceros)
    #print("Lower",vmin) # Condicion para excluir los negativos
    plt.imshow((np.log10(data)), vmin=np.nanpercentile(np.log10(data), permin), vmax=np.nanpercentile(np.log10(data), permax), origin = 'lower', cmap = cmapp)
    cbar = plt.colorbar()
    cbar.set_label(cbarname)
    plt.show()
    plt.close()

# Definimos las mismas funciones para plots pero añadiendo la posibilidad de que se puedan ajustar los límites de los ejes (sobre todo
# está pensado hacer esto al seleccionar las estrellas que no se pierda pista de qué píxeles se están seleccionando en los ejes X e Y, ya 
# que si no usando imshow sin más te empieza siempre en 0 (aunque tomes en X desde el píxel 2200 al 2500 te lo mostrará como de 0 a 300)

def show_data_with_axis(data, axislim, permin = 0.1, permax = 99.9, cmapp = 'bone', cbarname = ' ', plot_name = ''):
    #plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    extent=[axislim[0],axislim[1],axislim[2],axislim[3]]
    plt.imshow(data, vmin=np.nanpercentile(data,permin), vmax=np.nanpercentile(data,permax)
               , origin='lower' , cmap = cmapp,extent=extent)
    cbar = plt.colorbar()
    cbar.set_label(cbarname)
    plt.title(plot_name)
    plt.show()
    plt.close()


def show_data_log_with_axis(data_nor, axislim, permin = 8, permax = 99.9, cmapp = 'viridis', cbarname = ' ', plot_name = ''):
    data = np.copy(data_nor)
    # CAMBIO DE NEGATIVOS POR 0
    aneg = np.where(data<0)
    data[aneg] = 0
    # OFFSET
    aoffset = np.where(data>0)
    sin_ceros = data[aoffset]
    offset = np.nanmin(sin_ceros, axis = None)    
    data = data+offset #(Para evitar los ceros)
    #print("Lower",vmin) # Condicion para excluir los negativos
    extent=[axislim[0],axislim[1],axislim[2],axislim[3]]
    plt.figure(figsize=(10, 7))
    plt.imshow((np.log10(data)), vmin=np.nanpercentile(np.log10(data), permin), vmax=np.nanpercentile(np.log10(data), permax), origin = 'lower', cmap = cmapp,extent=extent)
    cbar = plt.colorbar()
    cbar.set_label(cbarname)
    plt.show()
    plt.close()
    
#FUNCIONES DE REDUCCIÓN DE DATOS

# Sacar datos del header para devolverlos con el formato que se usará después
def header_data(file, observatorio='CAHA'): 
    # Los datos que se obtienen del header van a estar definidos dentro de
    # una clase para poder acceder a todos con el mismo objeto sin tener que
    # estar cambiando los argumentos de salida de las funciones
    
    class Header_var:
        def __init__(self, im_size, data_cut, texp):
            self.im_size = im_size # Dimensiones de la imagen (x,y)
            self.data_cut = data_cut  # Límite (en pixeles) de la imagen de        
                      # ciencia para recortar el resto. Es una lista de cuatro 
                      # números. Ver abajo para ver el orden de los datos y 
                      # explicación de usos.
            self.texp = texp # Tiempo de exposición
            
    
    
    
    hdulist = fits.open(file) 
    # Importar los datos
    # header_full = hdulist[0].header
    # print(header_full)
    data = hdulist[0].data # Se quita el flip porque no queremos darles la vuelta
    # Importar información del header:
  
    # Tamaño de la imagen como tupla
    im_size = (int(hdulist[0].header['NAXIS2']), int(hdulist[0].header['NAXIS1']))
    #print(im_size)
    # Tiempo de exposición
    texp = int(hdulist[0].header['EXPTIME'])

    """
    Por lo visto cambia el formato de datos, no sé si de un observatorio a otro
    o de un telescopio a otro. Sería cuestión de configurar la función de recorte
    manualmente una vez que se tengan los datos buenos y se sepa cómo está el 
    formato del header.

    """
    if observatorio == 'CAHA':
        data_cut = hdulist[0].header['DATASEC'] # Límite en pixeles de las imágenes
        # Se quitan las llaves para que sólo con los números y caracteres de separación
        data_cut = data_cut[1:-1]
        
        x = data_cut.split(":")
        elem=[]
        for bla in x:
            elem.append(bla.split(","))
        # Se suma 1 porque van a ser límites para recortar arrays en python
        data_cut = np.array([int(elem[0][0]),int(elem[1][0])+1,
                             int(elem[0][1]), int(elem[1][1])+1])  
        # Formato de data cut:
        # El límite para recortar las imágenes se podrá escribir como:
        # imagen_data[data_cut[2]:data_cut[3], data_cut[0]:data_cut[1]]
  
    else:
        # Coge el tamaño de los datos y ya 
        data_cut = np.array([0,im_size[0], 0, im_size[1]])         
        
    hdulist.close
    output_header_data = Header_var(im_size, data_cut, texp)

    return output_header_data, data

    
def recortadora(data, data_cut, im_size, lim_data_big):   
    # data - Datos a recortar
    # data_cut - referencia en pixeles de la imagen
    # im_size - dimensión de la imagen
    # lim_data_big - referencia en pixeles de la imagen grande
    
    
    # Explicación (poner la pantalla para que se vea el rectángulo bien):
    """
    data_cut[0] = x1
    data_cut[1] = x2
    data_cut[2] = y1
    data_cut[3] = y2

  
     (i1,j1) __________________________________
            |                                  |
            |  (x1,y1) +--------------+(x2,y1) |
            |          |              |        |
            |          |  imagen      |        |  
            |  (x1,y1) +--------------+(x2,y2) | 
            |                                  |
            |______________bias________________|
        
     Se busca recortar los datos de una imagen más grande (bias, dark o flat)
     para que tenga el mismo tamaño que la imagen de ciencia. Como puede que 
     las imágenes grandes no empiecen por el pixel 0 se abre su header para 
     encontrar el equivalente a data_cut (que es para la imagen de ciencia).
     Para recortar se busca una expresión: x1 -> x2 , y1 -> y2      
     Al recortar datos en python de la forma matriz[lim1:lim2, lim3:lim4]
     se seleccionan primero las columnas entre lim1 y lim2-1 y después las filas
     entre lim3 y lim4-1. Por ello, nuestros índices al recortar serán:
         data[y1-j1 : (y1-j1)+ tamaño imagen dim 1 (y),
              x1-i1 : (x1-i1)+ tamaño imagen dim 0 (x)]
         
    
    """
    
    lim = data_cut - lim_data_big
    lim[1] = lim[0]+im_size[0]  # Limite superior en x
    lim[3] = lim[2]+im_size[1]  # Limite superior en y
    cut_data = data[lim[2]:lim[3],lim[0]:lim[1]]
    
    return cut_data



# Cálculo del bias
def median_bias(biasfiles, image_header, contrast = 2, num_sigma = 10,
                pintar_all = False, pintar_final = False, observatorio = 'CAHA',
                treat_bias = 1):
    """
    Abre todos los archivos de bias en la carpeta y devuelve la mediana de todas
    las imágenes en cada pixel -> Rayo iba a mirar estas cosas para mejorar lo de la mediana
    
    ARGUMENTOS DE ENTRADA:
        - biasfiles - Lista con todos los archivos en la carpeta del bias (Path)
        - image_header - Objeto de la clase Header_var (definida en el archivo
          de funfiones) que contiene la información obtenida del header de la
          imagen de ciencia ya con formato numérico (no str)
        - pintar_all - True para display de las imágenes de cada archivo
        - pintar_final - True para display de la imagen del bias combinado
        
    ARGUMENTOS DE SALIDA:
        - bias - Matriz con los datos del bias
        - nfiles - Número de archivos abiertos para crear la imagen final
           
    """
        
    bias_images = [] # Lista para acumular los datos del bias
    nfiles = 0 # Número de archivos que se abren
    for file in biasfiles:
        # Abrir los datos y la información relevante del header
        bias_header, data = header_data(file, observatorio)

        # Contar que ha abierto un archivo:
        nfiles = nfiles + 1
        # Sacar plots del bias
        if pintar_all == True:
            print('ARCHIVO:',file)
            print('t_exp: ' ,bias_header.texp,'s')

            show_data(data, cmapp = 'viridis',  cbarname = 'Numero de cuentas/pixel', plot_name = 'Imagen bias sin recortar',
                      permin=5, permax=95, )
        # Comprobar si hay que recortar
        if data.shape != image_header.im_size:
            # Recortar donde sabemos que está la imagen:           
            data = recortadora(data, image_header.data_cut, image_header.im_size, bias_header.data_cut)
            
        bias_images.append(data) 
      


    if nfiles == 0: # Si no hay datos, rellenar con ceros
        bias = np.zeros(image_header.im_size)
    else:
        bias = np.nanmedian(bias_images, axis=0)

    if pintar_final == True: # Sin tratar
            show_data(bias, cmapp = 'viridis', permin=5, permax=95, plot_name = 'Bias con medianas')  
     
    if treat_bias > 0:
        measured_map = np.copy(bias)  
        noise_variance_map = np.var(bias_images, axis=0) #estimador del error en cada pixel
    #Ahora, queremos comprobar si la señal es uniforme
        
        mean_measurement = np.nanmedian(measured_map) # estimador de la señal 'promedio'
        measured_variance = np.var(measured_map) #estimador de la varianza de la medida (señal+ruido)
        mean_noise_variance = np.nanmedian(noise_variance_map) #estimador del rudio 'promedio'
        signal_variance = measured_variance - mean_noise_variance #estimador de la varianza de la señal
        # dispersion = (np.percentile(bias,84)-np.percentile(bias,16))/2 # dispersión de una sigma, 
        # no recuerdo para qué era....
        
        if signal_variance > contrast*mean_noise_variance:
        # consideraremos que la varianza no es despreciable, de modo que se sustraerá el mapa entero 
        # puesto que no es estadísticamente uniforme
            return measured_map, noise_variance_map, nfiles
        else:
        # en este caso, se puede decir que el bias es prácticamente estidísitcamente unifore, de modo
        # que podremos restar la media (mediana)
        
            error_in_mean_measurement = np.sqrt(np.absolute(mean_noise_variance/len(noise_variance_map)+signal_variance))
            return mean_measurement*np.ones(image_header.im_size), error_in_mean_measurement, nfiles    
            
    else:
        return bias, np.var(bias), nfiles
        
        
   
#    return bias, nfiles





def flat_calc(flatsfiles, image_header, bias, master_dark, pintar_all = False, pintar_final = False, observatorio='CAHA'):
    if master_dark is None:
        master_dark = np.zeros(bias.shape)

    flats = []
    nfiles = 0
    # headers = []
    for file in flatsfiles:
        flat_header, data = header_data(file, observatorio)
        
        # Comprobar si hay que recortar
        if data.shape != image_header.im_size:
            # Recortar donde sabemos que está la imagen:           
            data = recortadora(data, image_header.data_cut, image_header.im_size, flat_header.data_cut)
        flat = data - bias - master_dark*flat_header.texp
        flats.append(flat/np.nanmedian(flat))
        # headers.append(hdulist[0].header)
        nfiles = nfiles + 1
        if pintar_all==True:
            print('ARCHIVO:',file)
            show_data(flat/np.nanmedian(flat), permin=5, permax=95, cmapp = 'viridis', cbarname='Factor del flat', plot_name='Flat abierto y recortado')
    
    if nfiles == 0:
        out_flat = np.zeros(image_header.im_size)
    else:
        out_flat = np.nanmedian(flats, axis=0)      
    if pintar_final==True:
        show_data(out_flat, permin=5, permax=95, cmapp = 'viridis', plot_name='Flat final',  cbarname='Factor del flat')        

    return out_flat, nfiles

def im_reduction(im_data, im_header, bias = None, flat = None, dark = None):
    if bias is None:
        bias = np.zeros(im_data.shape)
    if dark is None:
        dark = np.zeros(im_data.shape)  
    if flat is None:
        flat = np.ones(im_data.shape)
   # show_data(bias)
   # show_data(flat)
   # show_data(im_data)
    reduced_image = ((im_data-bias) - dark*im_header.texp)/(flat*im_header.texp)
    return reduced_image 

def cielo_4rec(reduced_image, rel_size, offset):
    """
    Se cogen cuatro cuadrados (o rectángulos si la imagen no fuera cuadrada)
    de la zona de las esquinas (1/rel_size*1/rel_size de la longitud total de la imagen 
    reducida y recortada. Se calcula la moda de cada zona (ahora sí) y se
    compara con la del resto de zonas usando la mediana.
    """
    

    zone = np.floor(np.asarray(reduced_image.shape)/rel_size) # Longitud de la zona
    offset = 100 # Número de pixeles que se aleja del borde

    # Primer rectángulo
    sky_1 = reduced_image[offset:offset+int(zone[0]),offset:offset+int(zone[1])]
    sky_1 = sky_1.flatten() # Convertir la matriz en array para...
    #sky_1p = stats.mode(sky_1) # ... calcular la moda
    #sky_1 = sky_1p[0] # Coger el primer término porque el segundo son las cuentas

    # Segundo rectángulo
    sky_2 = reduced_image[reduced_image.shape[0]-(offset+int(zone[0])):reduced_image.shape[0]-offset,
                      reduced_image.shape[1]-(offset+int(zone[1])):reduced_image.shape[1]-offset]
    sky_2 = sky_2.flatten()
    #sky_2p = stats.mode(sky_2)
    #sky_2 = sky_2p[0]   

    # Tercer rectángulo
    sky_3 = reduced_image[offset:offset+int(zone[0]),
                      reduced_image.shape[1]-(offset+int(zone[1])):reduced_image.shape[1]-offset]
    sky_3 = sky_3.flatten()
    #sky_3p = stats.mode(sky_3)
    #sky_3 = sky_3p[0]           

    # Cuarto rectángulo
    sky_4 = reduced_image[reduced_image.shape[0]-(offset+int(zone[0])):reduced_image.shape[0]-offset,
                      offset:offset+int(zone[1])]
    sky_4 = sky_4.flatten()
    #sky_4p = stats.mode(sky_4)
    #sky_4 = sky_4p[0] 

    # Calcular la mediana de los cuatro cuadrantes
    #sky_all = np.array([sky_1, sky_2, sky_3, sky_4])
    #sky = np.nanmedian(sky_all)
    all_skyes = np.concatenate((sky_1, sky_2, sky_3, sky_4), axis = 0)
    sky = stats.mode(all_skyes)
    sky = sky[0]
    return sky




# Adri:
def centroid(image, mask_cutoff=500):
    """
    image - matriz
    Si no funciona, probar a cambiar cutoff
    """
    image = np.array(image)
    mask = image > mask_cutoff
    image = np.ma.masked_array(image, mask=mask) 
    heigth, width = image.shape
    norm = np.sum(image)

    x = np.sum(np.sum(image, axis=0)*np.arange(width))
    y = np.sum(np.sum(image, axis=1)*np.arange(heigth))
    return int(np.round(y/norm)), int(np.round(x/norm))


def stackallnew(imagelist):
    """
    Conjunto de imágenes como lista (matrices)
    """
    #Calculamos los centroides de cada una de las imágenes
    x_c=[]
    y_c=[]
    for i in range (len(imagelist)):
        tempy,tempx = centroid(imagelist[i])
        x_c.append(tempx)
        y_c.append(tempy)
    #Calculamos el centroide "medio" al cual alinearemos todas las imágenes
    xc_mean = np.int(np.nanmean(x_c))
    yc_mean = np.int(np.nanmean(y_c))
   
    for i in range(len(x_c)):    
        x_c[i] = int(x_c[i])
        y_c[i] = int(y_c[i])
    x_c = np.array(x_c)
    y_c = np.array(y_c)
    diff_x = x_c - xc_mean
    diff_y = y_c - yc_mean
    new_imagelist = []
    
    wi_min = 1e19
    hi_min = 1e19
    for i in range(len(imagelist)):
        image = imagelist[i]
        hi , wi = image.shape
            
        if diff_x[i] >= 0:
            image = (image[0:(wi - diff_x[i]),:])
        else:
            image = (image[int(np.fabs(diff_x[i])):wi,:])
        
        if diff_y[i] >= 0:
            image = (image[:,0:(hi - diff_y[i])])
        else:
            image = (image[:,int(np.fabs(diff_y[i])):hi])    
        
        new_imagelist.append(image)
        hi , wi = image.shape
        if wi < wi_min:
            wi_min = wi
        if hi < hi_min:
            hi_min = hi
    
    for i in range(len(new_imagelist)):
        image = new_imagelist[i]
        image = image[0:hi_min,0:wi_min]
        new_imagelist[i] = image
    return np.nanmedian(new_imagelist, axis = 0)     

def mascara(img,width=18):
    
    m = len(img[0])
    n = len(img)
    
    min_array = np.zeros((n,m))
    for j in range(n):
        i = m - width
        val = 0
        indold = m
        while (i>width-1):
            minimum = np.argmin(img[j][i-width + 10:i+width - 10])
            ind = i-width + minimum
            min_array[j][ind:indold] = np.remainder(val,2)
            i = i - 2*width 
            val = val + 1
            indold = ind
        min_array[j][0:indold] = np.remainder(val,2)
    return min_array

def mask_detector(img,width = 18):
    
    
    shapei = img.shape
    m = shapei[1]
    n = shapei[0]
    
    min_array = np.zeros((n,m))
    paso = np.floor(np.linspace(0,n-1,20)).astype(int)
    for j in paso:
        i = m - width
        val = 1
        indold = i
        while (i>width-1):
            minimum = np.argmin(img[j][i-width:i+width])
            ind = i-width + minimum
            min_array[j][ind:indold] = 1-2*np.remainder(val,2)
            i = i - 2*width
            val = val + 1
            indold = ind
        min_array[j][0:indold] = 1-2*np.remainder(val,2)
        
    min_array = np.swapaxes(min_array,0,1)
        
    for i in range(m):
        val = np.sign(np.sum(min_array[i]))
        min_array[i] = val
        
    mask = np.swapaxes(min_array,0,1)
        
    return mask

def cocktail(img_0, img_18, mask):
    
    
    mask_1 = mask
    mask_2 = -mask
    mask_1[np.where(mask_1<0)] = 0
    mask_2[np.where(mask_2<0)] = 0
    
    R_img = img_0 * mask_1 + img_18 * mask_2
    L_img = img_0 * mask_2 + img_18 * mask_1
    
    return R_img, L_img

def mask_width(mask, row =100):
    vec = mask[row]
    ord_where = np.where(vec == 1)[0]
    exord_where = np.where(vec == -1)[0]
    choose_ord = ord_where[1:]-ord_where[:-1]
    choose_exord = exord_where[1:]-exord_where[:-1]
    ord_width = choose_ord[np.where(choose_ord>1)]
    exord_width = choose_exord[np.where(choose_exord>1)]
    width = int(round(np.mean(ord_width[2:-2])+np.mean(exord_width[2:-2])))
    return ord_width, exord_width, width   

def star_centroid(img, plot = False):
    
    #inputs: 
    
        #img = la imagen, pero con cuidao porque la imagen solo tiene que contener el objeto y sky!!
            #si en la imagen hay mas estrellas or whatever pues se puede recortar tipo:
            # imag[bla:bla, bla,bla] a ojo
        
        #plot = True te saca los plots de perfil de luminosidad de la estrella en funcion del radio
            #y RL frente a R^2 para ver si esta bien hecho el ajuste (explicado abajo)
            
    #outputs:
    
        #lum_objsky = te da la luminosidad (en las unidades del input) del objeto (sin quitar el cielo)
        
        #lum_pixelsky = te da la luminosidad de cada pixel de cielo (de cada pixel !!!!) osea la mediana vamos
        
        #lum_obj = te da la luminosidad del objeto quitado el cielo
    if(plot):    
        show_data(img)
    
    median = np.nanmedian(img)          #mediana pero teniendo en cuenta los valores de la estrella tambien (uego  calculamos mejor)
    
    xx = np.arange(img.shape[1])        #vector x
    yy = np.arange(img.shape[0])        #vector y
    
    weight = (img-median)               #el peso que le damos a la funcion centroide
    weight[weight<1] = 1                #en este caso ponemos a 1 lso pixeles por debajo de la mediana
    weight = weight**2                  #y damos un peso cuadrático a cada pixel (!! solo vale si es considerablemente mas brillante que el cielo)
    
    x_weight = xx*np.nansum(weight, axis=0)         #vector de x en el que cada elemento es x_i*sum(f(x = x_i, y)) (suma de todos los y)
    xcm = np.nansum(x_weight)/np.nansum(weight)     #xcm es el valor del centro de masas (para el peso que hayamos cogido)
    
    y_weight = yy*np.nansum(weight, axis=1)         #lo mismo que para x pero para y
    ycm = np.nansum(y_weight)/np.nansum(weight)
    
    r2 = ((xx-xcm)**2)[np.newaxis, :] + ((yy-ycm)**2)[:, np.newaxis]        #matriz que te da en cada pixel la distancia al centro de masas
    
    r2_object = r2[r2>10].flatten()[np.argmin(np.sqrt(r2[r2>10])*img[r2>10])]      
    
    #para calcular el radio del objeto lo que hacemos es buscar el mínimo de la función r*L (radio*luminosidad) y suponemos 
    #que ese es el punto en el que el objeto deja de ser brillante respecto al cielo
    #!!quitamos los elementos r2<10 porque a veces dan problemas (centro de masas no es necesariamente el punto mas brillante ej:estrella binaria ;) )
    
    sky = np.nanmedian(img[r2>r2_object])      #el valor de el cielo ahora lo calculamos para los puntos en los que solo tenemos cielo (r)
    
    if(plot):
        plt.figure( figsize=[12.8, 4.8])
        plt.subplot(121)
        plt.loglog(r2, img-sky,'k.', r2_object, img[r2==r2_object]-sky, 'r.')
        plt.xlabel('r^2 respecto al centro de masas')
        plt.ylabel('Luminosidad respecto al sky')
        plt.subplot(122)
        plt.loglog(r2, np.sqrt(r2)*img-sky, 'k.', r2_object, np.sqrt(r2_object)*img[r2==r2_object]-sky, 'r.')
        plt.xlabel('r^2 respecto al centro de masas')
        plt.ylabel('r * L')
        plt.show()
        
    r2_vec = r2.flatten()
    r_index = np.argsort(r2_vec)
    r2_vec = r2_vec[r_index]
    img_vec = img.flatten()[r_index] 
    lum_cum = np.cumsum(img_vec)
    img_obj_vec = img_vec - sky
    lum_nosky_cum = np.cumsum(img_obj_vec)
        
    
    lum_objsky = np.nansum(img[r2<r2_object])               #Luminosidad de los puntos de la estrella
    lum_pixelsky = np.nanmedian(img[r2>r2_object])          #Luminosidad de cada pixel de cielo (aka mediana)
    lum_obj = lum_objsky - lum_pixelsky*len(img[r2<r2_object])      #Luminosidad del objeto quitando el cielo

    if (plot):
        plt.figure()
        plt.plot(np.sqrt(r2_vec),lum_nosky_cum,'.k', np.sqrt(r2_object),lum_obj,'.r')
        plt.xlabel('r respecto al centro de masas')
        plt.ylabel('L_star tomando hasta el radio r')
        plt.show()
    
    return (lum_objsky, lum_pixelsky, lum_obj)
    

def star_luminosity_polarization(img, plot = True):
    
    #Este no lo explico porque solo se usa para polarizacion y rayo y yo ya sabemos como va
    
    ordinary_axis = img[475:525,450:500]
    extraordinary_axis = img[475:525,500:550]
    
    lum_ordinary, sky_ordinary, lum_star_ordinary = star_centroid(ordinary_axis, plot = plot)
    lum_extraordinary, sky_extraordinary, lum_star_extraordinary = star_centroid(extraordinary_axis, plot = plot)
    
    return lum_ordinary, sky_ordinary, lum_star_ordinary, lum_extraordinary, sky_extraordinary, lum_star_extraordinary

def normalized_flux_difference(angles, luminosity):
    
    #Solo polarizacion
    
    ind = np.argsort(angles)
    angles = angles[ind]
    luminosity = luminosity[ind]
    ord_extraord = np.resize(luminosity,(2,int(np.floor(len(luminosity)/2))))
    difference = (ord_extraord[0]-ord_extraord[1])/(ord_extraord[0]+ord_extraord[1])
    difference = np.array([difference,difference]).flatten()
    
    return angles, luminosity, difference

def normalized_flux_difference_2(angles, luminosity):
    
    #Solo polarizacion
    
    ind = np.argsort(angles)
    angles = angles[ind]
    luminosity = luminosity[ind]
    ord_extraord = np.resize(luminosity,(2,int(np.floor(len(luminosity)/2))))
    difference = (ord_extraord[0]-ord_extraord[1])/(ord_extraord[0]+ord_extraord[1])
    angles = angles[int(np.floor(len(luminosity)/2)):]
    luminosity = luminosity[int(np.floor(len(luminosity)/2)):]
    
    return angles, luminosity, difference

def histogram_img(img, values=60, xmin=750, xmax = 950):
    
    plt.hist(img,values,(xmin,xmax))
    
    return np.nanmedian(img)

def fourier_analysis(x, y, plot='False'):
    # OUTPUTS:
    #          a  : cosine coefficients, array
    #          b  : sine coefficients, array
    
    fourier_y = np.fft.rfft(y) / x.size
    fourier_y *= 2
    a = fourier_y.real
    b = -fourier_y.imag
    k_array = np.arange(a.size)
    x_new = np.linspace(x[0], x[-1], 101)
    power_spectrum = np.sqrt(a**2+b**2)
    reconstruccion = np.ones(x_new.size)*a[0]/2
    num = a.size
    length=x_new[-1]-x_new[0]
    i=1
    while i < num:
        reconstruccion += a[i]*np.cos(2*np.pi*i*x_new/length)+b[i]*np.sin(2*np.pi*i*x_new/length)
        i+=1
    if plot:
        plt.bar(k_array,power_spectrum)
        plt.show()
        plt.plot(x_new,reconstruccion)
        plt.show()


def stokes_parameters(flux_array):
    
    n = flux_array.size
    x = np.arange(n)
    Q = np.sum(flux_array*np.cos(np.pi*x/2))*2/n
    U = np.sum(flux_array*np.sin(np.pi*x/2))*2/n
    P = np.sqrt(Q**2+U**2) # polarization degree
    X = np.arctan(U/Q)/2   # polarization angle
    return P, X




