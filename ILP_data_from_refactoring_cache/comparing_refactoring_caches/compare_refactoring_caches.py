import zipfile
import csv
import io
import itertools
import os

def es_archivo_basura(nombre_archivo):
    if nombre_archivo.endswith('/'): return True
    if '__MACOSX' in nombre_archivo: return True
    if nombre_archivo.startswith('._') or '/._' in nombre_archivo: return True
    if nombre_archivo.endswith('.DS_Store'): return True
    if not nombre_archivo.lower().endswith('.csv'): return True
    return False

def comparar_csv_12_columnas(contenido1_bytes, contenido2_bytes):
    texto1 = contenido1_bytes.decode('utf-8', errors='replace')
    texto2 = contenido2_bytes.decode('utf-8', errors='replace')
    
    # Detectar si el separador es coma o punto y coma
    separador = ';' if ';' in texto1.splitlines()[0] else ','
    
    reader1 = csv.reader(io.StringIO(texto1), delimiter=separador)
    reader2 = csv.reader(io.StringIO(texto2), delimiter=separador)
    
    for fila1, fila2 in itertools.zip_longest(reader1, reader2):
        if fila1 is None or fila2 is None:
            return True
        if fila1[:12] != fila2[:12]:
            return True
            
    return False

def filtrar_zips_modificados(zip_original_path, zip_nuevo_path, zip_salida_path):
    with zipfile.ZipFile(zip_original_path, 'r') as zip1, \
         zipfile.ZipFile(zip_nuevo_path, 'r') as zip2, \
         zipfile.ZipFile(zip_salida_path, 'w', zipfile.ZIP_DEFLATED) as zip_salida:
        
        # 1. Crear un diccionario del ZIP 1 ignorando las carpetas
        # Guardará algo como: {"archivo.csv": "bytecode-viewer/archivo.csv"}
        mapa_zip1 = {}
        for ruta_zip1 in zip1.namelist():
            if not es_archivo_basura(ruta_zip1):
                nombre_base1 = os.path.basename(ruta_zip1)
                mapa_zip1[nombre_base1] = ruta_zip1
                
        cant_modificados = 0
        cant_nuevos = 0
        
        print("--- Procesando archivos ---")
        for ruta_zip2 in zip2.namelist():
            if es_archivo_basura(ruta_zip2):
                continue
            
            # Obtener solo el nombre del archivo, ej: "archivo.csv"
            nombre_base2 = os.path.basename(ruta_zip2)
            
            # 2. Comprobar si el nombre existe en el diccionario del ZIP 1
            if nombre_base2 in mapa_zip1:
                # Recuperar la ruta real dentro del ZIP 1
                ruta_real_zip1 = mapa_zip1[nombre_base2]
                
                contenido1 = zip1.read(ruta_real_zip1)
                contenido2 = zip2.read(ruta_zip2)
                
                # Comparar solo 12 columnas
                if comparar_csv_12_columnas(contenido1, contenido2):
                    # Guardamos el del zip 2 con su ruta original
                    zip_salida.writestr(ruta_zip2, contenido2)
                    cant_modificados += 1
                    print(f" [DIFERENTE] {nombre_base2}")
            else:
                # Si no está en el mapa del ZIP 1, es un archivo nuevo de verdad
                zip_salida.writestr(ruta_zip2, zip2.read(ruta_zip2))
                cant_nuevos += 1
                print(f" [NUEVO]     {nombre_base2}")
                
        cant_total = cant_modificados + cant_nuevos
        
        print("\n" + "="*45)
        print("          RESUMEN DEL PROCESO")
        print("="*45)
        print(f"  • Archivos nuevos (no existían en zip 1): {cant_nuevos}")
        print(f"  • Archivos con cambios (12 columnas):     {cant_modificados}")
        print(f"  • Total guardados en el ZIP final:        {cant_total}")
        print("="*45)
        print(f"Archivo generado en: '{zip_salida_path}'\n")


# --- Ejemplo de uso ---
if __name__ == "__main__":
    archivo_zip1 = "C:\\Users\\X1502\\Downloads\\refactoring_cache_files_Eclipse2026-06.zip"
    archivo_zip2 = "C:\\Users\\X1502\\Downloads\\refactoring_cache_CCR_updated.zip"
    archivo_salida = "csv_filtrados.zip" # El nombre del zip resultante
    
    filtrar_zips_modificados(archivo_zip1, archivo_zip2, archivo_salida)