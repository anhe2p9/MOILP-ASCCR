package com.refactoring.extractor.handlers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceDescription;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.jobs.Job;

public class SampleHandler extends AbstractHandler {

    @Override
    public Object execute(ExecutionEvent event) throws ExecutionException {
        // --- PARÁMETROS DE CONFIGURACIÓN ---
        String resultsBaseDir = "";
        String targetAlgo = "EpsilonConstraintAlgorithm";           // EpsilonConstraintAlgorithm o HybridMethodAlgorithm
        List<String> userPriority = Arrays.asList("loc", "extractions", "cc");
        String targetClass = "";                                    // Opcional: Nombre de clase (ej. "JSON.java" o "")                               // Opcional: Nombre de clase (ej. "JSON.java" o "")
        String projectSourceDir = ""; 								// Directorio padre o ZIP del proyecto general
        String targetSubmodule = "";
        
        // =================================================================
        // EJECUCIÓN EN SEGUNDO PLANO (JOB) PARA EVITAR QUE ECLIPSE SE CONGELE
        // =================================================================
        Job refactoringJob = new Job("Procesando Refactorizaciones MO-ILP") {
            @Override
            protected IStatus run(IProgressMonitor monitor) {
            	// =================================================================
                // 0. CONFIGURACIÓN DE RUTAS Y SISTEMA DE LOGS
                // =================================================================
                File projectSourceFile = new File(projectSourceDir);
                String originalName = projectSourceFile.getName();
                String cleanName = originalName.toLowerCase().endsWith(".zip") ? originalName.substring(0, originalName.length() - 4) : originalName;
                
                // Definir la carpeta destino ("..._refactored_...")
                File workingPath = new File(projectSourceFile.getParentFile(), cleanName + "_refactored_" + targetAlgo);
                
                // Limpiar la carpeta si existe de una ejecución anterior y crearla nueva
                if (workingPath.exists()) {
                    deleteDirectory(workingPath);
                }
                workingPath.mkdirs();

                // Generar nombre de archivo log dinámico
                String logFileName = cleanName;
                if (targetClass != null && !targetClass.trim().isEmpty()) {
                    String safeClassName = targetClass.replace(".java", "").replace("/", "_").replace("\\", "_");
                    logFileName += "_" + safeClassName;
                }
                logFileName += "_" + targetAlgo + "-v2026-06.log";
                
                // 🎯 CAMBIO 1: El Log se guarda dentro de la carpeta workingPath
                File logFile = new File(workingPath, logFileName);
                
                java.io.PrintStream originalOut = System.out;
                java.io.PrintStream originalErr = System.err;
                java.io.FileOutputStream fileOutputStream = null;
                
                boolean wasAutoBuilding = false; // Variable para recordar el estado original del compilador

                try {
                    fileOutputStream = new java.io.FileOutputStream(logFile, false); 
                    DualPrintStream dualOut = new DualPrintStream(originalOut, fileOutputStream);
                    DualPrintStream dualErr = new DualPrintStream(originalErr, fileOutputStream);
                    
                    System.setOut(dualOut);
                    System.setErr(dualErr);
                    
                    System.out.println("📝 Log de esta sesión guardado en: " + logFile.getAbsolutePath());
                    System.out.println("🚀 Iniciando procesamiento de refactorizaciones desde el Plug-in en segundo plano...");
                    long startTime = System.currentTimeMillis();
                    
                    IWorkspace workspace = ResourcesPlugin.getWorkspace();
                    IWorkspaceDescription desc = workspace.getDescription();
                    wasAutoBuilding = desc.isAutoBuilding(); // Guardamos el estado original
                    
                    if (wasAutoBuilding) {
                        desc.setAutoBuilding(false);
                        workspace.setDescription(desc);
                        System.out.println("🛑 Auto-build del workspace desactivado (todo build será explícito y controlado).");
                    }
                    
	                 // =================================================================
	                 // 1. DESCOMPRESIÓN / COPIA Y RESOLUCIÓN DE LA RAÍZ DEL PROYECTO
	                 // =================================================================
	
	                 // --- LIMPIEZA PREVIA EN WORKSPACE ---
	                 org.eclipse.core.resources.IProject oldProject = 
	                         org.eclipse.core.resources.ResourcesPlugin.getWorkspace().getRoot().getProject(cleanName);
	                 if (oldProject.exists()) {
	                     try {
	                         oldProject.delete(true, true, new org.eclipse.core.runtime.NullProgressMonitor());
	                         System.out.println("🧹 Proyecto previo '" + cleanName + "' purgado del workspace de Eclipse.");
	                     } catch (Exception e) {}
	                 }
	
	                 // COPIA O DESCOMPRESIÓN HACIA EL DIRECTORIO INDEPENDIENTE
	                 if (originalName.toLowerCase().endsWith(".zip")) {
	                     System.out.println("📦 Descomprimiendo '" + originalName + "' en '" + workingPath.getAbsolutePath() + "'...");
	                     unzip(projectSourceFile, workingPath);
	                 } else {
	                     System.out.println("📦 Copiando carpeta '" + originalName + "' a '" + workingPath.getAbsolutePath() + "' para no alterar el original...");
	                     try {
	                         copyDirectory(projectSourceFile, workingPath);
	                     } catch (IOException e) {
	                         System.err.println("❌ Error al copiar la carpeta del proyecto original.");
	                         e.printStackTrace();
	                     }
	                 }
	
	                 File foundTargetDir = ProjectUtils.findUnzippedProjectRoot(workingPath);
                     boolean isZipFile = originalName.toLowerCase().endsWith(".zip");
                     
                     // 🛠️ IGNORAR 'findUnzippedProjectRoot' si estamos copiando una carpeta directa
                     File projectRoot = (isZipFile && foundTargetDir != null) ? foundTargetDir : workingPath;
                     String projectName = cleanName;
                     String generalProjectName = cleanName;

                     // --- CORRECCIÓN: BÚSQUEDA PROFUNDA Y RECURSIVA DEL SUBMÓDULO REAL ---
                     if (targetSubmodule != null && !targetSubmodule.trim().isEmpty()) {
                         File directSubmodule = new File(projectRoot, targetSubmodule);
                         
                         if (directSubmodule.exists()) {
                             projectRoot = directSubmodule;
                         } else {
                             // Búsqueda recursiva robusta en todo el directorio de trabajo
                             File deepFound = findSubmoduleRecursively(workingPath, targetSubmodule);
                             if (deepFound != null) {
                                 projectRoot = deepFound;
                             } else {
                                 projectRoot = directSubmodule; // Fallback para visibilizar el error en consola
                             }
                         }
                         
                         projectName = targetSubmodule;
                         System.out.println("🎯 Enfocando refactorización en el submódulo: " + projectRoot.getAbsolutePath());
                     }
	
	                 // Purga adicional del proyecto/submódulo
	                 org.eclipse.core.resources.IProject oldSubProject = 
	                         org.eclipse.core.resources.ResourcesPlugin.getWorkspace().getRoot().getProject(projectRoot.getName());
	                 if (oldSubProject.exists()) {
	                     try {
	                         oldSubProject.delete(true, true, new org.eclipse.core.runtime.NullProgressMonitor());
	                     } catch (Exception e) {}
	                 }
                        
                    resolveProjectDependencies(projectRoot);
                    
                    // =================================================================
                    // LIMPIEZA PREVIA PARA EVITAR CONFLICTOS DE "CASE-SENSITIVITY"
                    // =================================================================
                    File virtualBinFolder = new File(projectRoot, "bin_eclipse_virtual");
                    if (virtualBinFolder.exists()) {
                        deleteDirectory(virtualBinFolder); 
                    }

                    File binFolder = new File(projectRoot, "bin");
                    if (binFolder.exists()) {
                        deleteDirectory(binFolder); 
                    }
                    
                    org.eclipse.core.resources.IProject staleProject =
                            org.eclipse.core.resources.ResourcesPlugin.getWorkspace()
                                    .getRoot().getProject(projectRoot.getName());
                    if (staleProject.exists()) {
                        try {
                            staleProject.delete(false, true, new org.eclipse.core.runtime.NullProgressMonitor());
                            System.out.println("🧹 Proyecto '" + projectRoot.getName() + "' desregistrado del workspace (estado previo descartado).");
                        } catch (org.eclipse.core.runtime.CoreException e) {
                            System.err.println("⚠️ No se pudo desregistrar el proyecto previo del workspace: " + e.getMessage());
                        }
                    }

	                 // =================================================================
	                 // 2. BUSCAR CARPETA DE RESULTADOS / CSV
	                 // =================================================================
	                 File targetResultsDir = resolveResultsDir(new File(resultsBaseDir), projectName, generalProjectName);
                    if (targetResultsDir == null || !targetResultsDir.exists()) {
                        System.err.println("⚠️ Error: No se encontró carpeta de resultados para '" + projectName + "' en:\n" + resultsBaseDir);
                        return Status.CANCEL_STATUS;
                    }

                    System.out.println("🔍 Explorando resultados en: " + targetResultsDir.getAbsolutePath());

                    // =================================================================
                    // 3. CARGAR Y PROCESAR RESULTADOS DESDE CSV
                    // =================================================================
                    Map<String, Map<String, List<ResultsProcessor.Extraction>>> resultsMap =
                            ResultsProcessor.processResults(targetResultsDir.getAbsolutePath(), targetAlgo, userPriority, targetClass);

                    int totalClases = resultsMap.size();
                    int totalMetodosOriginales = 0;
                    int totalExtracciones = 0;

                    for (Map<String, List<ResultsProcessor.Extraction>> metodosDeClase : resultsMap.values()) {
                        totalMetodosOriginales += metodosDeClase.size(); 
                        for (List<ResultsProcessor.Extraction> extracciones : metodosDeClase.values()) {
                            totalExtracciones += extracciones.size();
                        }
                    }

                    System.out.println("📊 Clases encontradas para modificar: " + totalClases);
                    System.out.println("🔍 Métodos originales a procesar: " + totalMetodosOriginales);
                    System.out.println("🛠️ Bloques de código totales a extraer: " + totalExtracciones);

                    // =================================================================
                    // 4. BÚSQUEDA DEL ARCHIVO JAVA Y APLICACIÓN
                    // =================================================================
                    int currentClassIndex = 0;
                    
                    for (Map.Entry<String, Map<String, List<ResultsProcessor.Extraction>>> classEntry : resultsMap.entrySet()) {
                        currentClassIndex++;
                        double progressPct = (currentClassIndex * 100.0) / totalClases;
                        String relativeClassPath = classEntry.getKey();
                        Map<String, List<ResultsProcessor.Extraction>> methodMap = classEntry.getValue();

                        File targetJavaFile = resolveJavaFile(projectRoot, relativeClassPath);

                        if (targetJavaFile == null || !targetJavaFile.exists()) {
                            System.err.println("👻 Archivo fantasma: No se encontró '" + relativeClassPath + "' en " + projectRoot.getName());
                            for (List<ResultsProcessor.Extraction> exts : methodMap.values()) {
                                for (ResultsProcessor.Extraction ext : exts) {
                                    ext.succeeded = false;
                                    ext.failReason = "Archivo Java no encontrado (Ghost file)";
                                }
                            }
                            continue;
                        }

                        System.out.println(String.format("\n📁 [%d/%d - %.1f%%] Procesando archivo: %s", 
                                currentClassIndex, totalClases, progressPct, targetJavaFile.getAbsolutePath()));

                        List<ResultsProcessor.Extraction> allClassExtractions = new ArrayList<>();
                        for (List<ResultsProcessor.Extraction> exts : methodMap.values()) {
                            allClassExtractions.addAll(exts);
                        }

                        try {
                            ASTModifier.processFileWithJFace(projectRoot, targetJavaFile, allClassExtractions);
                        } catch (Exception e) {
                            System.err.println("❌ Error procesando en " + targetJavaFile.getName());
                            e.printStackTrace();
                            for (ResultsProcessor.Extraction ext : allClassExtractions) {
                                ext.succeeded = false;
                                ext.failReason = "Excepción crítica durante modificación AST: " + e.getMessage();
                            }
                        }
                    }

                    // =================================================================
                    // 5. RESUMEN Y ESTADÍSTICAS FINALES DE EJECUCIÓN
                    // =================================================================
                    int totalExtractionsCount = 0;
                    int successfulExtractionsCount = 0;
                    int totalMethodsCount = 0;
                    int completedMethodsCount = 0;
                    int partialMethodsCount = 0;
                    int failedMethodsCount = 0;
                    int totalClassesCount = resultsMap.size();
                    int completedClassesCount = 0;
                    int partialClassesCount = 0;
                    int failedClassesCount = 0;

                    for (Map<String, List<ResultsProcessor.Extraction>> methodMap : resultsMap.values()) {
                        int classTotalExtractions = 0;
                        int classSuccessExtractions = 0;

                        for (List<ResultsProcessor.Extraction> extractions : methodMap.values()) {
                            totalMethodsCount++;
                            int methodTotal = extractions.size();
                            int methodSuccess = 0;

                            for (ResultsProcessor.Extraction ext : extractions) {
                                totalExtractionsCount++;
                                classTotalExtractions++;
                                if (ext.succeeded) {
                                    successfulExtractionsCount++;
                                    classSuccessExtractions++;
                                    methodSuccess++;
                                }
                            }

                            if (methodSuccess == methodTotal) {
                                completedMethodsCount++;
                            } else if (methodSuccess > 0) {
                                partialMethodsCount++;
                            } else {
                                failedMethodsCount++;
                            }
                        }

                        if (classTotalExtractions == 0) {
                            completedClassesCount++;
                        } else if (classSuccessExtractions == classTotalExtractions) {
                            completedClassesCount++;
                        } else if (classSuccessExtractions > 0) {
                            partialClassesCount++;
                        } else {
                            failedClassesCount++;
                        }
                    }

                    int failedExtractionsCount = totalExtractionsCount - successfulExtractionsCount;
                    double extSuccessPct = totalExtractionsCount > 0 ? (successfulExtractionsCount * 100.0 / totalExtractionsCount) : 0.0;
                    double extFailPct = totalExtractionsCount > 0 ? (failedExtractionsCount * 100.0 / totalExtractionsCount) : 0.0;
                    double methodCompPct = totalMethodsCount > 0 ? (completedMethodsCount * 100.0 / totalMethodsCount) : 0.0;
                    double methodPartPct = totalMethodsCount > 0 ? (partialMethodsCount * 100.0 / totalMethodsCount) : 0.0;
                    double methodFailPct = totalMethodsCount > 0 ? (failedMethodsCount * 100.0 / totalMethodsCount) : 0.0;
                    double classCompPct = totalClassesCount > 0 ? (completedClassesCount * 100.0 / totalClassesCount) : 0.0;
                    double classPartPct = totalClassesCount > 0 ? (partialClassesCount * 100.0 / totalClassesCount) : 0.0;
                    double classFailPct = totalClassesCount > 0 ? (failedClassesCount * 100.0 / totalClassesCount) : 0.0;
                    long elapsedTimeMs = System.currentTimeMillis() - startTime;
                    double elapsedTimeSec = elapsedTimeMs / 1000.0;

                    System.out.println("\n==================================================");
                    System.out.println("📊 RESUMEN FINAL DE LA EJECUCIÓN");
                    System.out.println("==================================================");
                    System.out.println(String.format("⏱️  Tiempo total de ejecución : %.2f segundos", elapsedTimeSec));
                    
                    System.out.println("\n✂️  EXTRACCIONES:");
                    System.out.println(String.format("   - Total planificadas : %d", totalExtractionsCount));
                    System.out.println(String.format("   - Exitosas           : %d (%.2f%%)", successfulExtractionsCount, extSuccessPct));
                    System.out.println(String.format("   - Fallidas           : %d (%.2f%%)", failedExtractionsCount, extFailPct));

                    System.out.println("\n🧩 MÉTODOS:");
                    System.out.println(String.format("   - Total procesados   : %d", totalMethodsCount));
                    System.out.println(String.format("   - Completados (100%%) : %d (%.2f%%)", completedMethodsCount, methodCompPct));
                    System.out.println(String.format("   - Parciales          : %d (%.2f%%)", partialMethodsCount, methodPartPct));
                    System.out.println(String.format("   - Fallidos (0%%)      : %d (%.2f%%)", failedMethodsCount, methodFailPct));

                    System.out.println("\n📁 CLASES:");
                    System.out.println(String.format("   - Total procesadas   : %d", totalClassesCount));
                    System.out.println(String.format("   - Completadas (100%%) : %d (%.2f%%)", completedClassesCount, classCompPct));
                    System.out.println(String.format("   - Parciales          : %d (%.2f%%)", partialClassesCount, classPartPct));
                    System.out.println(String.format("   - Fallidas (0%%)      : %d (%.2f%%)", failedClassesCount, classFailPct));
                    System.out.println("==================================================\n");
                    
                    // =================================================================
                    // 6. EXPORTAR CSV DE MOTIVOS DE RECHAZO
                    // =================================================================
                    File rejectedCsvFile = new File(workingPath, cleanName + "_" + targetAlgo + "_rejected_reasons_v2026-06.csv"); // El CSV se guarda dentro de workingPath
                    try (java.io.PrintWriter csvWriter = new java.io.PrintWriter(new java.io.FileWriter(rejectedCsvFile))) {
                        csvWriter.println("Clase;MetodoOriginal;OffsetInicio;Longitud;MotivoRechazo");
                        for (Map.Entry<String, Map<String, List<ResultsProcessor.Extraction>>> classEntry : resultsMap.entrySet()) {
                            String className = classEntry.getKey();
                            for (Map.Entry<String, List<ResultsProcessor.Extraction>> methodEntry : classEntry.getValue().entrySet()) {
                                String methodName = methodEntry.getKey();
                                for (ResultsProcessor.Extraction ext : methodEntry.getValue()) {
                                    if (!ext.succeeded) {
                                        try {
                                            int offset = (ext.range != null && ext.range.length > 0) ? ext.range[0] : -1;
                                            int length = (ext.range != null && ext.range.length > 1) ? (ext.range[1] - ext.range[0]) : -1;
                                            String safeReason = (ext.failReason == null || ext.failReason.isEmpty()) 
                                                                ? "Saltado (posicion invalida o fallo de snapping)" 
                                                                : ext.failReason.replace(";", ",").replace("\"", "'").replace("\n", " ").replace("\r", "");
                                            csvWriter.println(className + ";" + methodName + ";" + offset + ";" + length + ";\"" + safeReason + "\"");
                                        } catch (Exception innerE) {
                                            System.err.println("⚠️ Aviso: No se pudo escribir la fila de la extracción fallida en " + methodName + ". Motivo: " + innerE.getMessage());
                                        }
                                    }
                                }
                            }
                        }
                        System.out.println("💾 Archivo CSV con motivos de rechazo exportado en:\n   " + rejectedCsvFile.getAbsolutePath());
                    } catch (Exception e) {
                        System.err.println("⚠️ Error crítico al abrir o guardar el CSV de rechazos: " + e.getMessage());
                    }
                    
                    // =================================================================
                    // 7. LIMPIEZA DE ESTRUCTURA (RESTAURAR ESTADO ORIGINAL)
                    // =================================================================
                    System.out.println("🧹 Limpiando artefactos temporales del compilador JDT...");
                    if (projectRoot != null && projectRoot.exists()) {
                        File[] children = projectRoot.listFiles();
                        if (children != null) {
                            for (File f : children) {
                                if (f.isDirectory() && f.getName().startsWith("bin_eclipse_virtual")) {
                                    deleteDirectory(f);
                                }
                            }
                        }
                        deleteDirectory(new File(projectRoot, "bin")); 
                        new File(projectRoot, ".project").delete();
                        new File(projectRoot, ".classpath").delete();
                        deleteDirectory(new File(projectRoot, ".settings"));
                    }

                    System.out.println("🎉 ¡Proceso completado con éxito!");

                } catch (Exception e) {
                    System.err.println("❌ Error crítico durante la ejecución:");
                    e.printStackTrace();
                    return Status.CANCEL_STATUS;
                } finally {
                    // =================================================================
                    // RESTAURAR LA CONSOLA Y EL AUTO-BUILD (¡Muy importante!)
                    // =================================================================
                    System.setOut(originalOut);
                    System.setErr(originalErr);
                    if (fileOutputStream != null) {
                        try {
                            fileOutputStream.close();
                        } catch (java.io.IOException e) {}
                    }
                    
                    // Restauración garantizada del Auto-build si estaba activado previamente
                    try {
                        IWorkspace workspace = ResourcesPlugin.getWorkspace();
                        IWorkspaceDescription desc = workspace.getDescription();
                        if (desc.isAutoBuilding() != wasAutoBuilding) {
                            desc.setAutoBuilding(wasAutoBuilding);
                            workspace.setDescription(desc);
                            // Este print sale por la consola de verdad de Eclipse, pues ya se restauró en la línea anterior
                            System.out.println("🔄 Auto-build restaurado a su estado original: " + wasAutoBuilding);
                        }
                    } catch (Exception e) {
                        // Silenciar error en restauración
                    }
                }
                
                return Status.OK_STATUS;
            }
        };

        // Planificar y ejecutar el Job (devuelve el control de la UI de Eclipse inmediatamente)
        refactoringJob.schedule();

        return null;
    }

    // =================================================================
    // MÉTODOS AUXILIARES (INTACTOS, CON PEQUEÑO ARREGLO EN MAVEN)
    // =================================================================

    private static File resolveJavaFile(File projectRoot, String relativeClassPath) {
        File directFile = new File(projectRoot, relativeClassPath);
        if (directFile.exists()) {
            return directFile;
        }

        String normalizedTarget = relativeClassPath.replace('\\', '/');
        String targetFileName = new File(normalizedTarget).getName();

        List<File> candidates = new ArrayList<>();
        searchFilesByName(projectRoot, targetFileName, candidates);

        if (candidates.isEmpty()) {
            return null;
        }

        File bestCandidate = null;
        int maxMatchLength = -1;

        for (File candidate : candidates) {
            String candidatePath = candidate.getAbsolutePath().replace('\\', '/');
            
            if (candidatePath.endsWith(normalizedTarget)) {
                return candidate; 
            }

            String[] targetParts = normalizedTarget.split("/");
            int matchCount = 0;
            for (int i = targetParts.length - 1; i >= 0; i--) {
                if (candidatePath.contains(targetParts[i])) {
                    matchCount++;
                } else {
                    break;
                }
            }

            if (matchCount > maxMatchLength) {
                maxMatchLength = matchCount;
                bestCandidate = candidate;
            }
        }

        return bestCandidate;
    }

    private static void searchFilesByName(File dir, String fileName, List<File> result) {
        File[] files = dir.listFiles();
        if (files == null) return;
        for (File f : files) {
            if (f.isDirectory()) {
                searchFilesByName(f, fileName, result);
            } else if (f.getName().equals(fileName)) {
                result.add(f);
            }
        }
    }

    private static File resolveResultsDir(File resultsBaseDir, String primaryName, String fallbackName) {
        File targetDir = findResults(resultsBaseDir, primaryName);
        
        // Fallback: si no encuentra la carpeta con el nombre del submódulo, prueba con el general
        if (targetDir == null && fallbackName != null && !fallbackName.isEmpty()) {
            System.out.println("⚠️ No se encontró carpeta de resultados para '" + primaryName + "', buscando el proyecto general '" + fallbackName + "'...");
            targetDir = findResults(resultsBaseDir, fallbackName);
        }
        return targetDir;
    }

    private static File findResults(File baseDir, String name) {
        String targetName = name.toLowerCase();
        File projectDir = null;

        if (baseDir.getName().toLowerCase().contains(targetName)) {
            projectDir = baseDir;
        } else {
            projectDir = findFolderByName(baseDir, targetName);
        }

        if (projectDir == null) return null;

        File[] subFiles = projectDir.listFiles();
        if (subFiles != null) {
            for (File f : subFiles) {
                if (f.isDirectory() && f.getName().toLowerCase().contains("3-objective")) {
                    return f;
                }
            }
        }
        return projectDir;
    }

    private static File findFolderByName(File dir, String folderName) {
        File[] files = dir.listFiles();
        if (files == null) return null;
        for (File f : files) {
            if (f.isDirectory()) {
                if (f.getName().toLowerCase().contains(folderName)) {
                    return f;
                }
                File subMatch = findFolderByName(f, folderName);
                if (subMatch != null) return subMatch;
            }
        }
        return null;
    }

    private static void unzip(File zipFile, File destDir) throws IOException {
        // No borramos el destDir aquí porque lo hemos limpiado
        // al principio de run() y ahora contiene nuestro archivo .log abierto.
        if (!destDir.exists()) {
            destDir.mkdirs();
        }

        byte[] buffer = new byte[1024];
        // ... (el resto del método se queda exactamente igual)
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile))) {
            ZipEntry zipEntry = zis.getNextEntry();
            while (zipEntry != null) {
                File newFile = newFile(destDir, zipEntry);
                if (zipEntry.isDirectory()) {
                    newFile.mkdirs();
                } else {
                    File parent = newFile.getParentFile();
                    if (!parent.exists()) {
                        parent.mkdirs();
                    }
                    try (FileOutputStream fos = new FileOutputStream(newFile)) {
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                }
                zipEntry = zis.getNextEntry();
            }
            zis.closeEntry();
        }
    }
    
    private static void copyDirectory(File source, File target) throws IOException {
        if (source.isDirectory()) {
            if (!target.exists()) {
                target.mkdirs();
            }
            String[] children = source.list();
            if (children != null) {
                for (String child : children) {
                    copyDirectory(new File(source, child), new File(target, child));
                }
            }
        } else {
            java.nio.file.Files.copy(source.toPath(), target.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }
    
    private static void resolveProjectDependencies(File projectRoot) {
        File pomFile = new File(projectRoot, "pom.xml");
        
        if (pomFile.exists()) {
            System.out.println("⚙️ Proyecto Maven detectado. Verificando y descargando dependencias...");
            
            File mvnw = new File(projectRoot, "mvnw.cmd"); 
            String mavenCmd = mvnw.exists() ? mvnw.getAbsolutePath() : "mvn";
            String absoluteDependencyDir = new File(projectRoot, "target/dependency").getAbsolutePath();

            String baseMvnArgs = mavenCmd + " dependency:copy-dependencies -DincludeScope=test -DoutputDirectory=\"" + absoluteDependencyDir + "\"";
            String fullCmd;
            
            if (System.getProperty("os.name").toLowerCase().contains("win")) {
                fullCmd = "cmd.exe /c " + baseMvnArgs;
            } else {
                fullCmd = baseMvnArgs;
            }
            
            boolean success = runMavenAndPatchIfNeeded(fullCmd, projectRoot.getAbsolutePath());
            
            if (success) {
                System.out.println("✅ Librerías Maven descargadas/parcheadas correctamente en 'target/dependency'.");
            } else {
                System.err.println("⚠️ Fallo al descargar dependencias de Maven tras intentar parchear.");
            }
        } else {
            System.out.println("ℹ️ No se detectó 'pom.xml'. Se asume proyecto Ant o estructura estándar.");
        }
    }

    private static File newFile(File destinationDir, ZipEntry zipEntry) throws IOException {
        File destFile = new File(destinationDir, zipEntry.getName());
        String destDirPath = destinationDir.getCanonicalPath();
        String destFilePath = destFile.getCanonicalPath();
        if (!destFilePath.startsWith(destDirPath + File.separator)) {
            throw new IOException("Entrada ZIP fuera del directorio destino: " + zipEntry.getName());
        }
        return destFile;
    }

    private static void deleteDirectory(File dir) {
        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                if (f.isDirectory()) deleteDirectory(f);
                else f.delete();
            }
        }
        dir.delete();
    }
    
    private static boolean runMavenAndPatchIfNeeded(String mvnArgs, String projectRootPath) {
        int maxRetries = 2; 
        int attempt = 0;

        while (attempt < maxRetries) {
            attempt++;
            System.out.println("🚀 Ejecutando Maven (Intento " + attempt + ")...");
            
            try {
                ProcessBuilder pb = new ProcessBuilder();
                if (System.getProperty("os.name").toLowerCase().contains("win")) {
                    String command = mvnArgs.startsWith("cmd.exe /c ") ? mvnArgs.substring(11) : mvnArgs;
                    pb.command("cmd.exe", "/c", command);
                } else {
                    pb.command("sh", "-c", mvnArgs);
                }
                pb.directory(new File(projectRootPath));
                
                // ===== CAMBIO VITAL: Redirigir Errores al Input Stream =====
                pb.redirectErrorStream(true); 

                Process process = pb.start();
                
                java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()));
                StringBuilder mavenOutput = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    mavenOutput.append(line).append("\n");
                    System.out.println("   [Maven] " + line); 
                }
                
                int exitCode = process.waitFor();
                
                if (exitCode == 0) {
                    return true; 
                }
                
                System.out.println("⚠️ Maven detectó un error. Analizando salida para auto-parcheo...");
                
                java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("([\\w\\.-]+):([\\w\\.-]+):[\\w\\.-]+:([\\w\\.-]+)(?: was not found| \\(compile\\))");
                java.util.regex.Matcher matcher = pattern.matcher(mavenOutput.toString());
                
                boolean patchedSomething = false;
                File pomFile = new File(projectRootPath, "pom.xml");

                while (matcher.find()) {
                    String errorArtifactId = matcher.group(2);
                    System.out.println("🔍 Detectada dependencia rota: " + errorArtifactId);

                    if (errorArtifactId.contains("jadx")) {
                        System.out.println("🛠️ Aplicando parche conocido para JADX...");
                        PomPatcher.autoPatchDependency(pomFile, errorArtifactId, "io.github.skylot", "1.4.7");
                        patchedSomething = true;
                    }
                }

                if (!patchedSomething) {
                    System.out.println("❌ No se encontró un parche automático para los errores devueltos.");
                    return false; 
                }

            } catch (Exception e) {
                System.err.println("❌ Excepción al ejecutar el parcheo de Maven:");
                e.printStackTrace();
                return false;
            }
        }
        
        return false; 
    }
    
    private static File findSubmoduleRecursively(File dir, String submoduleName) {
        if (dir == null || !dir.isDirectory()) return null;
        
        File[] files = dir.listFiles();
        if (files == null) return null;
        
        // Priorizar el nivel actual
        for (File f : files) {
            if (f.isDirectory() && f.getName().equals(submoduleName)) {
                return f;
            }
        }
        
        // Buscar en profundidad
        for (File f : files) {
            if (f.isDirectory()) {
                File found = findSubmoduleRecursively(f, submoduleName);
                if (found != null) return found;
            }
        }
        
        return null;
    }
    
    // =================================================================
    // CLASE AUXILIAR PARA EL LOG (DUPLICADOR DE SALIDA)
    // =================================================================
    private static class DualPrintStream extends java.io.PrintStream {
        private final java.io.OutputStream fileOut;

        public DualPrintStream(java.io.PrintStream consoleOut, java.io.OutputStream fileOut) {
            super(consoleOut, true);
            this.fileOut = fileOut;
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            super.write(buf, off, len); 
            try {
                fileOut.write(buf, off, len); 
            } catch (java.io.IOException e) {
            }
        }

        @Override
        public void write(int b) {
            super.write(b);
            try {
                fileOut.write(b);
            } catch (java.io.IOException e) {}
        }
        
        @Override
        public void flush() {
            super.flush();
            try {
                fileOut.flush();
            } catch (java.io.IOException e) {}
        }
    }
}