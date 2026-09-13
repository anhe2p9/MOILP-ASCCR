package com.refactoring.extractor.handlers;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MavenClasspathResolver {

    /**
     * Resuelve el classpath exacto usando Maven nativo, soportando multi-módulos.
     * Devuelve un array listo para ASTParser.setEnvironment()
     */
    public static String[] resolveClasspath(File projectDirectory) throws Exception {
        // 1. Usamos un archivo temporal absoluto del sistema. 
        // Esto evita que los sub-módulos escriban en rutas relativas distintas.
        Path outputFilePath = Files.createTempFile("jdt-classpath-", ".txt");
        
        List<String> command = new ArrayList<>();
        command.add(System.getProperty("os.name").toLowerCase().contains("win") ? "mvn.cmd" : "mvn");
        
        // OPCIONAL PERO RECOMENDADO: Si un módulo depende de otro en el mismo proyecto,
        // Maven necesita compilar para resolverlo. Puedes añadir "test-compile" antes:
        // command.add("test-compile"); 
        
        command.add("dependency:build-classpath");
        command.add("-Dmdep.outputFile=" + outputFilePath.toAbsolutePath().toString());
        // CLAVE PARA MULTI-MÓDULO: Le dice a Maven que no sobrescriba el archivo por cada submódulo
        command.add("-Dmdep.appendOutput=true"); 
        command.add("-DincludeScope=test");
        
        // En proyectos masivos, a veces un submódulo falla. Esto fuerza a Maven 
        // a seguir resolviendo el resto de módulos y sacar todas las dependencias posibles.
        command.add("--fail-at-end"); 

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(projectDirectory);
        
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        
        Process process = pb.start();
        int exitCode = process.waitFor();
        
        if (exitCode != 0) {
            System.err.println("⚠️ Advertencia: Maven devolvió error " + exitCode + ". Es común en multi-módulos complejos. Se intentará leer lo que se haya resuelto.");
        }

        if (!Files.exists(outputFilePath)) {
            return new String[0]; 
        }
        
        // 2. Maven escribirá múltiples líneas en el archivo (una por módulo)
        List<String> lines = Files.readAllLines(outputFilePath);
        
        // Limpiamos
        Files.deleteIfExists(outputFilePath);

        // 3. Usamos un Set para eliminar todos los JARs duplicados entre submódulos
        Set<String> uniqueJars = new HashSet<>();
        for (String line : lines) {
            if (line != null && !line.trim().isEmpty()) {
                String[] jars = line.trim().split(File.pathSeparator);
                uniqueJars.addAll(Arrays.asList(jars));
            }
        }
        
        return uniqueJars.toArray(new String[0]);
    }
}