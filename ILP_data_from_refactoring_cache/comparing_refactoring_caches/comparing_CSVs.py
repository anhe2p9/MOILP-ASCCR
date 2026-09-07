import zipfile
import csv
import io

def inspeccionar_csv(zip1_path, zip2_path, nombre_archivo_buscar):
    with zipfile.ZipFile(zip1_path, 'r') as z1, zipfile.ZipFile(zip2_path, 'r') as z2:
        # Buscar el archivo dentro de los zips (sin importar la ruta exacta)
        path1 = next((f for f in z1.namelist() if nombre_archivo_buscar in f and '__MACOSX' not in f), None)
        path2 = next((f for f in z2.namelist() if nombre_archivo_buscar in f and '__MACOSX' not in f), None)

        if not path1 or not path2:
            print(f"❌ No se encontró el archivo '{nombre_archivo_buscar}' en alguno de los ZIPs.")
            print(f"   En ZIP 1: {path1}")
            print(f"   En ZIP 2: {path2}")
            return

        print(f"🔍 Comparando:\n 1️⃣ {path1}\n 2️⃣ {path2}\n")

        # Leer archivos
        t1 = z1.read(path1).decode('utf-8', errors='replace')
        t2 = z2.read(path2).decode('utf-8', errors='replace')

        # Detectar posible separador (coma o punto y coma)
        separador = ';' if ';' in t1.splitlines()[0] else ','
        print(f"ℹ️ Separador detectado: '{separador}'")

        reader1 = list(csv.reader(io.StringIO(t1), delimiter=separador))
        reader2 = list(csv.reader(io.StringIO(t2), delimiter=separador))

        print(f"📊 Filas totales -> ZIP 1: {len(reader1)} | ZIP 2: {len(reader2)}\n")

        diferencias = 0
        max_mostrar = 5  # Solo mostramos las primeras 5 diferencias

        min_filas = min(len(reader1), len(reader2))
        for i in range(min_filas):
            row1_12 = reader1[i][:12]
            row2_12 = reader2[i][:12]

            if row1_12 != row2_12:
                diferencias += 1
                if diferencias <= max_mostrar:
                    print(f"⚠️ DIFERENCIA EN FILA {i + 1}:")
                    print(f"   ZIP 1 (cols 1-12): {row1_12}")
                    print(f"   ZIP 2 (cols 1-12): {row2_12}")
                    print("-" * 50)

        if len(reader1) != len(reader2):
            print(f"⚠️ Diferencia en número de filas: ZIP 1 tiene {len(reader1)} y ZIP 2 tiene {len(reader2)}.")

        if diferencias == 0 and len(reader1) == len(reader2):
            print("✅ ¡Las primeras 12 columnas son EXACTAMENTE IGUALES en todas las filas!")
        else:
            print(f"\n❌ Se encontraron {diferencias} filas con diferencias en las primeras 12 columnas.")

# --- Ejecutar inspección ---
if __name__ == "__main__":
    zip1 = "C:\\Users\\X1502\\Downloads\\refactoring_cache_files_Eclipse2026-06.zip"
    zip2 = "C:\\Users\\X1502\\Downloads\\refactoring_cache_CCR_updated.zip"

    # Pon aquí solo una parte del nombre de cualquier CSV que quieras inspeccionar
    csv_a_probar = "bytecode-viewer@src.main.java-the.bytecode.club.bytecodeviewer.api-ASMResourceUtil.java-renameClassNode-146.csv" 
    
    inspeccionar_csv(zip1, zip2, csv_a_probar)