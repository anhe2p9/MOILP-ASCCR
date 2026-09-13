package com.refactoring.extractor.handlers;

import java.io.File;

public class ProjectUtils {

    /**
     * Búsqueda ASCENDENTE: Para encontrar el proyecto raíz dado un archivo .java específico.
     * Soporta correctamente arquitecturas Maven Multimodular y proyectos Ant sin salirse de los límites.
     */
    public static File findProjectRoot(File file) {
        File current = file.isDirectory() ? file : file.getParentFile();
        File rootCandidate = null;

        while (current != null) {
            boolean hasPom = new File(current, "pom.xml").exists();
            boolean hasProject = new File(current, ".project").exists();
            boolean hasSrc = new File(current, "src").exists();
            boolean hasBuildXml = new File(current, "build.xml").exists();

            if (hasPom || hasProject || hasSrc || hasBuildXml) {
                rootCandidate = current;
                
                if (hasPom) {
                    // MAVEN MULTIMODULAR: Seguimos subiendo SOLO si el directorio padre también tiene un pom.xml
                    // En el momento en que un padre ya no tiene pom.xml, hemos llegado a la raíz real.
                    File parent = current.getParentFile();
                    if (parent == null || !new File(parent, "pom.xml").exists()) {
                        break; 
                    }
                } else {
                    // ANT / ESTÁNDAR: Nos quedamos con la primera raíz legítima que encontremos.
                    // Si siguiéramos subiendo, correríamos el riesgo de salirnos al workspace del usuario.
                    break;
                }
            }
            current = current.getParentFile();
        }
        
        return rootCandidate != null ? rootCandidate : (file.isDirectory() ? file : file.getParentFile());
    }

    /**
     * Búsqueda DESCENDENTE: Para usar después de descomprimir un ZIP.
     * Busca el directorio raíz real dentro del envoltorio (wrapper) generado por la extracción.
     */
    public static File findUnzippedProjectRoot(File dir) {
        // ¿La propia carpeta donde hemos descomprimido ya es la raíz?
        if (new File(dir, "pom.xml").exists() || new File(dir, "build.xml").exists() || new File(dir, "src").exists()) {
            return dir;
        }
        
        // Si no, exploramos el primer nivel de subcarpetas (comportamiento típico de ZIPs de GitHub)
        File[] children = dir.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory()) {
                    if (new File(child, "pom.xml").exists() || new File(child, "build.xml").exists() || new File(child, "src").exists()) {
                        return child;
                    }
                }
            }
        }
        return dir; // Fallback
    }
}