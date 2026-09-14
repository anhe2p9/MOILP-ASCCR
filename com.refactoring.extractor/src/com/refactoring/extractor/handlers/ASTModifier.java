package com.refactoring.extractor.handlers;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Map;


import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IFolder;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectDescription;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRoot;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.Path;
import org.eclipse.jdt.core.IClasspathEntry;
import org.eclipse.jdt.core.ICompilationUnit;
import org.eclipse.jdt.core.IJavaProject;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.compiler.IProblem;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.Block;
import org.eclipse.jdt.core.dom.Modifier;
import org.eclipse.jdt.core.dom.NodeFinder;
import org.eclipse.jdt.core.dom.Statement;
import org.eclipse.jdt.internal.corext.refactoring.code.ExtractMethodRefactoring;
import org.eclipse.jdt.launching.JavaRuntime;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.Position;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.TextChange;
import org.eclipse.text.edits.TextEdit;
import org.eclipse.text.edits.UndoEdit;
import org.eclipse.jdt.launching.IVMInstall;
import org.eclipse.jdt.launching.IVMInstallType;

@SuppressWarnings("restriction")
public class ASTModifier {

    private static final boolean DEBUG_MODE = true;

    public static void processFileWithJFace(File projectRoot, File javaFile, List<ResultsProcessor.Extraction> extractions) throws Exception {
        if (extractions == null || extractions.isEmpty()) return;

        ICompilationUnit cu = getCompilationUnitForFile(projectRoot, javaFile);
        if (cu == null) {
            System.err.println("❌ No se pudo mapear el archivo Java al modelo de JDT de Eclipse: " + javaFile.getAbsolutePath());
            return;
        }

        // 1. ACTIVAR WORKING COPY (Indispensable para refactorizaciones JDT)
        cu.becomeWorkingCopy(new NullProgressMonitor());

        try {
            String source = cu.getSource();

            // 2. ORDENAR EXTRACCIONES ESTRICTAMENTE DE ABAJO HACIA ARRIBA
            extractions.sort((a, b) -> {
                int offsetCmp = Integer.compare(b.range[0], a.range[0]);
                if (offsetCmp != 0) return offsetCmp;
                return Integer.compare(b.range[1] - b.range[0], a.range[1] - a.range[0]);
            });

            Document document = new Document(source);
            String category = "EXTRACTIONS_CATEGORY";
            document.addPositionCategory(category);
        
        document.addPositionUpdater(new org.eclipse.jface.text.IPositionUpdater() {
            @Override
            public void update(org.eclipse.jface.text.DocumentEvent event) {
                int eventOffset = event.getOffset();
                int eventOldLength = event.getLength();
                int eventNewLength = (event.getText() == null) ? 0 : event.getText().length();
                int delta = eventNewLength - eventOldLength;

                try {
                    org.eclipse.jface.text.Position[] categoryPositions = event.getDocument().getPositions(category);
                    for (org.eclipse.jface.text.Position pos : categoryPositions) {
                        if (pos.isDeleted()) continue;
                        
                        int posOffset = pos.getOffset();
                        int posEnd = posOffset + pos.getLength();
                        
                        // 1. Actualizar el INICIO del bloque (newOffset)
                        int newOffset = posOffset;
                        if (eventOffset < posOffset) {
                            if (eventOffset + eventOldLength <= posOffset) {
                                newOffset += delta; 
                            } else {
                                newOffset = eventOffset + eventNewLength; // Solapamiento
                            }
                        // } else if (eventOffset == posOffset && eventOldLength == 0) {
                            // Si JDT inserta código EXACTAMENTE al inicio del bloque, empujamos el bloque hacia abajo
                            // newOffset += delta;
                        }
                        
                        // 2. Actualizar el FINAL del bloque (newEnd)
                        int newEnd = posEnd;
                        // CAUSANTE SOLUCIONADO: Usar <= en lugar de <. 
                        // Si JDT inserta el nuevo método exactamente en la frontera final (10939 <= 10939),
                        // el bloque padre engullirá la llamada al método correctamente.
                        if (eventOffset <= posEnd) {
                            newEnd += delta;
                        }
                        
                        // 3. Aplicar los desplazamientos automáticos
                        pos.setOffset(newOffset);
                        pos.setLength(Math.max(0, newEnd - newOffset));
                    }
                } catch (org.eclipse.jface.text.BadPositionCategoryException e) {
                    // Ignorar silenciosamente
                }
            }
        });

        List<Position> positions = new ArrayList<>();
        for (ResultsProcessor.Extraction ext : extractions) {
            int off = ext.range[0];
            int len = ext.range[1] - ext.range[0];

            if (off < 0 || off + len > document.getLength()) {
                positions.add(null);
                continue;
            }

            Position pos = new Position(off, len);
            document.addPosition(category, pos);
            positions.add(pos);
        }

        // 3.5. LÍNEA BASE DE ERRORES DE COMPILACIÓN
        Set<String> baselineErrorSignatures = getCompileErrorSignatures(cu);

        java.util.Map<String, Integer> methodCounters = new java.util.HashMap<>();
        
        // --- NUEVO: Agrupar por el método de origen (originalMethodName) ---
        java.util.Map<String, List<Integer>> methodGroups = new java.util.LinkedHashMap<>();
        for (int i = 0; i < extractions.size(); i++) {
            String origName = extractions.get(i).originalMethodName != null ? extractions.get(i).originalMethodName : "unknown";
            methodGroups.computeIfAbsent(origName, k -> new ArrayList<>()).add(i);
        }

        // Bucle 1: Iteramos por grupo (cada método original)
        for (java.util.Map.Entry<String, List<Integer>> groupEntry : methodGroups.entrySet()) {
            List<Integer> extIndices = groupEntry.getValue();
            
            // Backup del estado ANTES de empezar con ESTE método específico
            String preGroupDocState = document.get();
            int[] preGroupOffsets = new int[positions.size()];
            int[] preGroupLengths = new int[positions.size()];
            for (int j = 0; j < positions.size(); j++) {
                if (positions.get(j) != null) {
                    preGroupOffsets[j] = positions.get(j).getOffset();
                    preGroupLengths[j] = positions.get(j).getLength();
                }
            }

            boolean retryTopToBottom = false;

            // Bucle 2: Sistema de reintentos
            for (int attempt = 0; attempt < 2; attempt++) {
                if (attempt == 1) {
                    if (!retryTopToBottom) break; // Si no hay error, no hay segundo intento
                    
                    // Restaurar el código fuente al estado previo a este método
                    document.set(preGroupDocState);
                    cu.getBuffer().setContents(preGroupDocState);
                    
                    // Restaurar las posiciones de TODAS las extracciones globales
                    for (int j = 0; j < positions.size(); j++) {
                        if (positions.get(j) != null) {
                            positions.get(j).undelete(); // Revive la posición si JFace la marcó como borrada
                            positions.get(j).setOffset(preGroupOffsets[j]);
                            positions.get(j).setLength(preGroupLengths[j]);
                        }
                    }
                    cu.reconcile(AST.JLS25, true, null, new NullProgressMonitor());

                    // Limpiar estado y reordenar ESTE GRUPO de arriba hacia abajo
                    for (int idx : extIndices) {
                        extractions.get(idx).succeeded = false;
                        extractions.get(idx).failReason = null;
                    }
                    extIndices.sort((idxA, idxB) -> Integer.compare(extractions.get(idxA).range[0], extractions.get(idxB).range[0]));
                    
                    System.out.println("🔄 Conflicto de variables detectado en '" + groupEntry.getKey() + "'. Localizando las variables dentro del bloque y reintentando...");
                    
                    // --- NUEVO: APLICAMOS LA TRANSFORMACIÓN AST ---
                    try {
                        localizeVariablesInDocument(cu, document, positions, extIndices);
                    } catch (Exception e) {
                        System.err.println("⚠️ Fallo interno intentando pre-procesar las variables del bucle: " + e.getMessage());
                        e.printStackTrace(System.err);
                    }
                }

                boolean groupHasDuplicateError = false;

                // Bucle 3: Procesar las extracciones de este grupo
                for (int i : extIndices) {
                    ResultsProcessor.Extraction ext = extractions.get(i);
                    Position pos = positions.get(i);

                    if (pos == null || pos.isDeleted() || pos.getLength() <= 0) {
                        continue;
                    }

            // 🆕 Medición de tiempo/offsets/nombre de ESTA extracción concreta, para el CSV de tiempos.
            // Se usan arrays de tamaño 1 como "contenedores mutables" para poder actualizarlos dentro
            // del try y seguir leyéndolos en el finally, pase lo que pase (éxito, fallo, o abandono
            // para reintentar tras un conflicto de variables).
            long extractionStartTime = System.currentTimeMillis();
            int[] loggedOffsets = { pos.getOffset(), pos.getOffset() + pos.getLength() };
            String[] loggedName = { null };

            try {
            int currentOffset = pos.getOffset();
            int currentLength = pos.getLength();
            int requestedOffset = currentOffset;
            int requestedLength = currentLength;

            String origName = ext.originalMethodName != null ? ext.originalMethodName : "unknown_method";
            int parenIndex = origName.indexOf('(');
            if (parenIndex != -1) {
                origName = origName.substring(0, parenIndex);
            }

            String cleanOrigName = origName.replaceAll("[^a-zA-Z0-9_]", "");
            if (cleanOrigName.isEmpty()) {
                cleanOrigName = "extractedMethod";
            } else {
                cleanOrigName = Character.toLowerCase(cleanOrigName.charAt(0)) + cleanOrigName.substring(1);
                if (Character.isDigit(cleanOrigName.charAt(0))) {
                    cleanOrigName = "m" + cleanOrigName;
                }
            }

            int count = methodCounters.getOrDefault(cleanOrigName, 1);
            String newMethodName = cleanOrigName + "Extracted" + count;
            methodCounters.put(cleanOrigName, count + 1);
            loggedName[0] = newMethodName;

            // =================================================================
            // INICIO: PASO 1 - SELECCIÓN SEMÁNTICA BASADA EN NODOS AST
            // =================================================================
            // 1. FORZAR SINCRONIZACIÓN ANTES DE CREAR EL AST
            cu.reconcile(AST.JLS25, ICompilationUnit.FORCE_PROBLEM_DETECTION | ICompilationUnit.ENABLE_BINDINGS_RECOVERY, null, new NullProgressMonitor());

            // 2. SELECCIÓN SEMÁNTICA BASADA EN NODOS AST
            ASTParser parser = ASTParser.newParser(AST.JLS25);
            parser.setSource(cu); 
            parser.setResolveBindings(true);
            parser.setStatementsRecovery(true);
            parser.setBindingsRecovery(true);
            // ⚠️ ELIMINADO: parser.setProject() y parser.setUnitName() para evitar desvincular el typeRoot

            org.eclipse.jdt.core.dom.CompilationUnit astCu = 
                (org.eclipse.jdt.core.dom.CompilationUnit) parser.createAST(new NullProgressMonitor());

            // 3. Ajuste semántico de rango
            int[] semanticRange = selectSemanticASTNodes(astCu, currentOffset, currentLength);

            if (semanticRange == null) {
                System.err.println("⏭️ Saltando '" + newMethodName + "' en " + javaFile.getName()
                        + ": No se encontraron nodos AST válidos.");
                ext.failReason = "Selección AST semántica no encontró sentencias válidas";
                ext.succeeded = false;
                continue;
            }

            currentOffset = semanticRange[0];
            currentLength = semanticRange[1];
            loggedOffsets[0] = currentOffset;
            loggedOffsets[1] = currentOffset + currentLength;
            int deltaStart = Math.abs(currentOffset - requestedOffset);
            int deltaEnd = Math.abs((currentOffset + currentLength) - (requestedOffset + requestedLength));
            if (deltaStart > 5 || deltaEnd > 5) {
                System.out.println("🧭 Ajuste semántico de AST en '" + newMethodName + "' (" + javaFile.getName() + "): "
                        + "pedido=[" + requestedOffset + "," + (requestedOffset + requestedLength) + "] "
                        + "→ ajustado AST=[" + currentOffset + "," + (currentOffset + currentLength) + "]");
            }

            if (DEBUG_MODE) {
                try {
                    String selectedText = document.get(currentOffset, currentLength);
                    System.out.println("\n🕵️ DEBUG RANGO AST - " + newMethodName + " en " + javaFile.getName());
                    System.out.println("Texto de nodos AST: [" + selectedText + "]");
                } catch (Exception e) {
                    System.out.println("Error leyendo el documento en offset: " + currentOffset);
                    e.printStackTrace(System.err);
                }
            }

            try {
                // ---- DIAGNÓSTICO EXTRA ----
                // IJavaProject jp = cu.getJavaProject();
                // System.out.println("🔬 JavaProject: " + jp.getElementName());
                // System.out.println("🔬 astCu.getJavaElement() == cu ? " + (astCu.getJavaElement() == cu)
                //        + " | astCu.getJavaElement(): " + astCu.getJavaElement());

                // org.eclipse.jdt.core.IType objectType = jp.findType("java.lang.Object");
                // System.out.println("🔬 Resolución de java.lang.Object: " + objectType
                //        + (objectType != null ? " | Fuente: " + objectType.getCompilationUnit() + " / " + objectType.getClassFile() : ""));

                // org.eclipse.jdt.core.IClasspathEntry[] resolved = jp.getResolvedClasspath(true);
                // for (org.eclipse.jdt.core.IClasspathEntry e : resolved) {
                    // System.out.println("   📦 CP: " + e.getEntryKind() + " -> " + e.getPath());
                // }
                // System.out.println("🔬 Compliance del proyecto: " + jp.getOption(JavaCore.COMPILER_COMPLIANCE, true));
                // ---- FIN DIAGNÓSTICO ----

                cu.reconcile(AST.JLS25, ICompilationUnit.FORCE_PROBLEM_DETECTION | ICompilationUnit.ENABLE_BINDINGS_RECOVERY, null, new NullProgressMonitor());
                ExtractMethodRefactoring refactoring = new ExtractMethodRefactoring(astCu, currentOffset, currentLength);
                
                RefactoringStatus status = refactoring.checkInitialConditions(new NullProgressMonitor());
                
                
                if (status.hasFatalError()) {
                    System.err.println("❌ InitialConditions falló para '" + newMethodName + "': " + Arrays.toString(status.getEntries()));
                } else {
                    refactoring.setMethodName(newMethodName);
                    refactoring.setVisibility(Modifier.PRIVATE);
                    refactoring.setReplaceDuplicates(false);

                    try {
                        RefactoringStatus finalStatus = refactoring.checkFinalConditions(new NullProgressMonitor());
                        status.merge(finalStatus);
                    } catch (Throwable e) {
                        System.err.println("==================================================");
                        System.err.println("💥 DETALLE DEL ERROR INTERNO EN JDT ('" + newMethodName + "')");
                        System.err.println("Mensaje: " + e.getMessage());
                        System.err.println("Clase de excepción: " + e.getClass().getName());
                        System.err.println("Estado CU - isWorkingCopy: " + cu.isWorkingCopy() + " | hasUnsavedChanges: " + cu.hasUnsavedChanges());
                        System.err.println("Longitud buffer CU: " + (cu.getBuffer() != null ? cu.getBuffer().getLength() : "null"));
                        System.err.println("Traza completa de la excepción:");
                        e.printStackTrace(System.err);
                        System.err.println("==================================================");

                        ext.failReason = "Excepción en checkFinalConditions (" + e.getClass().getSimpleName() + "): " + e.getMessage();
                        ext.succeeded = false;
                        continue;
                    }
                }

                Change change = refactoring.createChange(new NullProgressMonitor());
                TextChange textChange = null;
                if (change instanceof TextChange) {
                    textChange = (TextChange) change;
                } else if (change instanceof org.eclipse.ltk.core.refactoring.CompositeChange) {
                    for (Change child : ((org.eclipse.ltk.core.refactoring.CompositeChange) change).getChildren()) {
                        if (child instanceof TextChange) {
                            textChange = (TextChange) child;
                            break;
                        }
                    }
                }

                if (textChange != null) {
                    TextEdit edit = textChange.getEdit();
                    
                    int[] backupOffsets = new int[positions.size()];
                    int[] backupLengths = new int[positions.size()];
                    for (int j = 0; j < positions.size(); j++) {
                        if (positions.get(j) != null) {
                            backupOffsets[j] = positions.get(j).getOffset();
                            backupLengths[j] = positions.get(j).getLength();
                        }
                    }
                    
                    UndoEdit undo = edit.apply(document, TextEdit.CREATE_UNDO | TextEdit.UPDATE_REGIONS);
                    cu.getBuffer().setContents(document.get());

                    Set<String> newErrorSignatures = getCompileErrorSignatures(cu);
                    newErrorSignatures.removeAll(baselineErrorSignatures);

                    if (!newErrorSignatures.isEmpty()) {
                        undo.apply(document);
                        cu.getBuffer().setContents(document.get());
                        
                        for (int j = 0; j < positions.size(); j++) {
                            if (positions.get(j) != null) {
                                positions.get(j).setOffset(backupOffsets[j]);
                                positions.get(j).setLength(backupLengths[j]);
                            }
                        }
                        
                        cu.reconcile(AST.JLS25, true, null, new NullProgressMonitor());

                        ext.succeeded = false;
                        ext.failReason = "Introdujo errores de compilación: " + newErrorSignatures.toString();
                        
                        // 1. Detectar si el error es de variables (duplicadas, no inicializadas o no resueltas)
                        boolean hasVariableError = newErrorSignatures.toString().contains("Duplicate local variable") 
                                                || newErrorSignatures.toString().contains("may not have been initialized")
                                                || newErrorSignatures.toString().contains("cannot be resolved to a variable")
                                                || newErrorSignatures.toString().matches(".*\\b\\w+ cannot be resolved\\b.*");

                        // 2. Disparar el arreglo de AST solo en el primer intento si hay fallo de variables
                        if (attempt == 0 && hasVariableError) {
                            groupHasDuplicateError = true;
                            break; // Abortamos para preparar el código e intentar de nuevo
                        }
                        
                        // 3. Imprimir el mensaje de revertido ÚNICAMENTE si ya fracasó nuestro último intento o no era un error reparable
                        if (attempt == 1 || !hasVariableError) {
                            System.err.println("↩️ Revertido definitivamente '" + newMethodName + "' en " + javaFile.getName()
                            + ": se deja original. Error intratable → " + newErrorSignatures);
                        }
                        
                        continue;
		            }
                    
                    cu.reconcile(AST.JLS25, true, null, new NullProgressMonitor());
                    ext.succeeded = true;
                    
                } else {
                    System.err.println("❌ No se pudo extraer un TextChange válido para '" + newMethodName + "'");
                    ext.succeeded = false;
                    ext.failReason = "No se pudo generar un TextChange válido";
                }

	            } catch (Exception e) {
	                System.err.println("❌ Error al extraer método '" + newMethodName + "' en " + javaFile.getName() + ": " + e.getMessage());
	                e.printStackTrace(System.err);
	                ext.succeeded = false;
	                ext.failReason = "Excepción interna: " + e.getMessage();
	            }
            } finally {
                // 🆕 Se registra SIEMPRE, sea cual sea la salida de esta iteración (éxito, fallo,
                // o abandono para reintentar tras un conflicto de variables detectado más arriba).
                ext.executionTimeMs = System.currentTimeMillis() - extractionStartTime;
                ext.extractedMethodName = loggedName[0];
                ext.appliedStartOffset = loggedOffsets[0];
                ext.appliedEndOffset = loggedOffsets[1];
            }
	        } // Fin del bucle 3 (for int i : extIndices)
	        
	        // --- NUEVO: Control del bucle de intentos ---
	        if (groupHasDuplicateError) {
	            retryTopToBottom = true;
	        } else {
	            break; // El intento fue exitoso (o falló por otras causas irreparables), rompemos y pasamos al siguiente método
	        }
	    } // Fin del bucle 2 (for attempt)
	} // Fin del bucle 1 (for groupEntry)
	// ------------------------------------------------
	
    cu.commitWorkingCopy(true, new NullProgressMonitor());
    } finally {
        // LIBERAR WORKING COPY AL FINALIZAR
        cu.discardWorkingCopy();
    }
	}

    // =====================================================================
    // IMPLEMENTACIÓN PASO 1: SELECCIÓN SEMÁNTICA DE NODOS AST
    // =====================================================================
    /**
     * Revisa semánticamente el AST en busca de nodos de tipo Statement que se 
     * encuentren dentro o alineados con los offsets especificados. 
     * Garantiza el envío de límites exactos de nodos AST completos a la API de refactorización.
     * 
     * @param astRoot Raíz de la CompilationUnit en el AST.
     * @param offset  Offset de inicio del texto.
     * @param length  Longitud del fragmento seleccionado.
     * @return int[] con {startOffset, exactLength} o null si no abarca nodos válidos.
     */
    private static int[] selectSemanticASTNodes(ASTNode astRoot, int offset, int length) {
        ASTNode node = NodeFinder.perform(astRoot, offset, length);
        if (node == null) {
            return null;
        }

        // Caso A: El nodo encontrado es una única sentencia concreta (ej. IfStatement, ExpressionStatement, etc.)
        if (node instanceof Statement && !(node instanceof Block)) {
            return new int[] { node.getStartPosition(), node.getLength() };
        }

        // Caso B: El nodo es un contenedor de varias sentencias (Block)
        if (node instanceof Block) {
            Block block = (Block) node;
            List<Statement> containedStatements = new ArrayList<>();

            int reqStart = offset;
            int reqEnd = offset + length;

            for (Object obj : block.statements()) {
                if (obj instanceof Statement) {
                    Statement stmt = (Statement) obj;
                    int stmtStart = stmt.getStartPosition();
                    int stmtEnd = stmtStart + stmt.getLength();

                    // Criterio de inclusión semántica: la sentencia tiene intersección real con la selección
                    boolean isInside = (stmtStart >= reqStart && stmtEnd <= reqEnd) ||
                                       (stmtStart <= reqStart && stmtEnd > reqStart) ||
                                       (stmtStart < reqEnd && stmtEnd >= reqEnd);

                    if (isInside) {
                        containedStatements.add(stmt);
                    }
                }
            }

            if (!containedStatements.isEmpty()) {
                Statement first = containedStatements.get(0);
                Statement last = containedStatements.get(containedStatements.size() - 1);

                int semanticStart = first.getStartPosition();
                int semanticEnd = last.getStartPosition() + last.getLength();
                return new int[] { semanticStart, semanticEnd - semanticStart };
            }
        }

        // Caso C: Fallback semántico - Búsqueda de la sentencia o bloque ancestro más cercano
        ASTNode parent = node;
        while (parent != null && !(parent instanceof Statement)) {
            parent = parent.getParent();
        }

        if (parent instanceof Statement) {
            return new int[] { parent.getStartPosition(), parent.getLength() };
        }

        return null;
    }

    // =====================================================================
    // VERIFICACIÓN DE COMPILACIÓN (Rollback Automático)
    // =====================================================================
    private static Set<String> getCompileErrorSignatures(ICompilationUnit cu) throws Exception {
        ASTParser parser = ASTParser.newParser(AST.JLS25);
        parser.setSource(cu);
        parser.setResolveBindings(true);
//        parser.setProject(cu.getJavaProject());
//        parser.setUnitName(cu.getPath().toString());
        parser.setStatementsRecovery(true);
        parser.setBindingsRecovery(true);

        org.eclipse.jdt.core.dom.CompilationUnit astCu =
                (org.eclipse.jdt.core.dom.CompilationUnit) parser.createAST(new NullProgressMonitor());

        Set<String> signatures = new HashSet<>();
        for (IProblem problem : astCu.getProblems()) {
            if (problem.isError()) {
                signatures.add(problem.getID() + "::" + problem.getMessage());
            }
        }
        return signatures;
    }

    public static void logProjectBaselineErrors(File projectRoot) {
        try {
            File anyJavaFile = findAnyJavaFile(projectRoot);
            if (anyJavaFile == null) return;

            ICompilationUnit cu = getCompilationUnitForFile(projectRoot, anyJavaFile);
            if (cu == null) return;

            IProject project = cu.getJavaProject().getProject();
            project.build(org.eclipse.core.resources.IncrementalProjectBuilder.FULL_BUILD, new NullProgressMonitor());

            org.eclipse.core.resources.IMarker[] markers = project.findMarkers(
                    org.eclipse.core.resources.IMarker.PROBLEM, true,
                    org.eclipse.core.resources.IResource.DEPTH_INFINITE);

            long errorCount = Arrays.stream(markers)
                    .filter(m -> m.getAttribute(org.eclipse.core.resources.IMarker.SEVERITY,
                            org.eclipse.core.resources.IMarker.SEVERITY_INFO)
                            == org.eclipse.core.resources.IMarker.SEVERITY_ERROR)
                    .count();

            System.out.println("📐 LÍNEA BASE del proyecto: " + errorCount + " error(es) de compilación previos.");
        } catch (Exception e) {
            System.err.println("⚠️ No se pudo calcular la línea base de errores: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    private static File findAnyJavaFile(File dir) {
        File[] children = dir.listFiles();
        if (children == null) return null;
        for (File f : children) {
            if (f.isFile() && f.getName().endsWith(".java")) {
                return f;
            }
        }
        for (File f : children) {
            if (f.isDirectory()) {
                File found = findAnyJavaFile(f);
                if (found != null) return found;
            }
        }
        return null;
    }

    private static ICompilationUnit getCompilationUnitForFile(File projectRoot, File javaFile) throws CoreException {
        IWorkspace workspace = ResourcesPlugin.getWorkspace();
        IWorkspaceRoot root = workspace.getRoot();

        // Elimina la llamada a ProjectUtils.findProjectRoot(javaFile) y usa directamente el argumento:
        File projectDir = projectRoot;
        if (projectDir == null || !projectDir.exists()) return null;

        String projectName = projectDir.getName();
        IProject project = root.getProject(projectName);

        if (!project.exists()) {
            IProjectDescription description = workspace.newProjectDescription(projectName);
            description.setLocation(Path.fromOSString(projectDir.getAbsolutePath()));
            project.create(description, new NullProgressMonitor());
            project.open(new NullProgressMonitor());
        } else if (!project.isOpen()) {
            project.open(new NullProgressMonitor());
        }

        if (!project.hasNature(JavaCore.NATURE_ID)) {
            IProjectDescription desc = project.getDescription();
            desc.setNatureIds(new String[] { JavaCore.NATURE_ID });
            project.setDescription(desc, new NullProgressMonitor());
        }

        IJavaProject javaProject = JavaCore.create(project);
        setupJavaProjectClasspath(javaProject, project);

        IPath filePath = Path.fromOSString(javaFile.getAbsolutePath());
        IFile fileInWorkspace = project.getFile(filePath.makeRelativeTo(Path.fromOSString(projectDir.getAbsolutePath())));
        
        // Forzar sincronización entre disco y workspace de Eclipse
        fileInWorkspace.refreshLocal(org.eclipse.core.resources.IResource.DEPTH_ZERO, new NullProgressMonitor());

        return JavaCore.createCompilationUnitFrom(fileInWorkspace);
    }

    private static void setupJavaProjectClasspath(IJavaProject javaProject, IProject project) throws CoreException {
        File projectRootFolder = project.getLocation().toFile();
        File pomFile = new File(projectRootFolder, "pom.xml");

        List<IClasspathEntry> cleanEntries = new ArrayList<>();

        // 1. JRE: resolver una VM concreta y verificada, NO "default" a ciegas
        IVMInstall vm = resolveMatchingJRE(pomFile);
        cleanEntries.add(JavaCore.newContainerEntry(JavaRuntime.newJREContainerPath(vm)));

        // 2. Carpeta(s) fuente reales del submódulo
        findAndAddSourceFolders(projectRootFolder, project, cleanEntries);
        if (cleanEntries.stream().noneMatch(e -> e.getEntryKind() == IClasspathEntry.CPE_SOURCE)) {
            cleanEntries.add(JavaCore.newSourceEntry(project.getFullPath()));
        }

        // 3. Classpath de dependencias: pedírselo a Maven, no reconstruirlo a mano
        if (pomFile.exists()) {
            List<String> mvnClasspath = resolveClasspathViaMavenCli(projectRootFolder);
            Set<String> seenArtifactKeys = new HashSet<>(); // para deduplicar por groupId:artifactId
            for (String jarPathStr : mvnClasspath) {
                File jarFile = new File(jarPathStr);
                if (!jarFile.exists() || !jarFile.getName().endsWith(".jar")) continue;
                if (isJarConflictivoConJRE(jarFile.getName())) continue; // Ignorar JARs XML legacy que pisan el módulo java.xml del JRE
                
                IPath jarPath = Path.fromOSString(jarFile.getAbsolutePath());
                if (seenArtifactKeys.add(jarFile.getName())) { // dedup simple por nombre de fichero
                    cleanEntries.add(JavaCore.newLibraryEntry(jarPath, null, null));
                }
            }
        } else {
            // Fallback: tu escaneo original, solo si no hay pom.xml
            int[] counters = new int[]{0, 0};
            findAndAddLibraryJars(projectRootFolder, cleanEntries, counters);
        }

        javaProject.setRawClasspath(cleanEntries.toArray(new IClasspathEntry[0]), new NullProgressMonitor());

        // 4. Compliance: el mismo que declara el pom, no un VERSION_11 fijo
        Map<String, String> options = javaProject.getOptions(true);
        String compliance = detectComplianceFromPom(pomFile); // p.ej. "17", "21"...
        JavaCore.setComplianceOptions(compliance, options);
        javaProject.setOptions(options);

        project.refreshLocal(org.eclipse.core.resources.IResource.DEPTH_INFINITE, new NullProgressMonitor());
    }
    
    private static List<String> resolveClasspathViaMavenCli(File moduleDir) throws CoreException {
        try {
            File outputFile = File.createTempFile("cp_", ".txt");
            outputFile.deleteOnExit();

            String mvnCmd = System.getProperty("os.name").toLowerCase().contains("win") ? "mvn.cmd" : "mvn";
            ProcessBuilder pb = new ProcessBuilder(
                    mvnCmd, "-q", "-DincludeScope=test",
                    "-Dmdep.outputFile=" + outputFile.getAbsolutePath(),
                    "dependency:build-classpath"
            );
            pb.directory(moduleDir);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // Volcamos la salida por si Maven falla, para poder diagnosticarlo
            StringBuilder mvnOutput = new StringBuilder();
            try (java.io.BufferedReader reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) mvnOutput.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("⚠️ 'mvn dependency:build-classpath' falló (exit " + exitCode + "):\n" + mvnOutput);
                return java.util.Collections.emptyList();
            }

            String cpLine = new String(java.nio.file.Files.readAllBytes(outputFile.toPath())).trim();
            String separator = System.getProperty("path.separator"); // ';' en Windows, ':' en Unix
            return Arrays.asList(cpLine.split(java.util.regex.Pattern.quote(separator)));

        } catch (Exception e) {
            System.err.println("❌ Error invocando Maven para resolver el classpath: " + e.getMessage());
            e.printStackTrace(System.err);
            return java.util.Collections.emptyList();
        }
    }
    
    private static String detectComplianceFromPom(File pomFile) {
        try {
            if (!pomFile.exists()) return JavaCore.VERSION_17; // valor por defecto razonable
            String content = new String(java.nio.file.Files.readAllBytes(pomFile.toPath()));
            java.util.regex.Matcher m = java.util.regex.Pattern
                    .compile("<maven\\.compiler\\.release>\\s*(\\d+)\\s*</maven\\.compiler\\.release>")
                    .matcher(content);
            if (m.find()) return m.group(1);
            m = java.util.regex.Pattern
                    .compile("<maven\\.compiler\\.target>\\s*([\\d.]+)\\s*</maven\\.compiler\\.target>")
                    .matcher(content);
            if (m.find()) return m.group(1).replace("1.", ""); // "1.8" -> "8"
            return JavaCore.VERSION_17;
        } catch (Exception e) {
            return JavaCore.VERSION_17;
        }
    }

    private static IVMInstall resolveMatchingJRE(File pomFile) {
        String compliance = detectComplianceFromPom(pomFile);
        for (IVMInstallType type : JavaRuntime.getVMInstallTypes()) {
            for (IVMInstall vm : type.getVMInstalls()) {
                if (vm.getInstallLocation() != null
                        && vm.getInstallLocation().getName().contains(compliance)) {
                    return vm;
                }
            }
        }
        IVMInstall def = JavaRuntime.getDefaultVMInstall();
        System.out.println("⚠️ No se encontró una JDK " + compliance + " instalada; usando la VM por defecto: "
                + (def != null ? def.getInstallLocation() : "NINGUNA"));
        return def;
    }

    private static final String[] JARS_CONFLICTIVOS_CON_JRE = {
        "xml-apis", "xercesimpl", "xalan", "serializer", "xml-resolver"
    };

    private static boolean isJarConflictivoConJRE(String jarFileName) {
        String lower = jarFileName.toLowerCase();
        for (String sospechoso : JARS_CONFLICTIVOS_CON_JRE) {
            if (lower.contains(sospechoso)) return true;
        }
        return false;
    }
    
    private static void findAndAddLibraryJars(File currentDir, List<IClasspathEntry> entries, int[] counters) {
        if (currentDir == null || !currentDir.isDirectory()) return;
        
        String name = currentDir.getName().toLowerCase();
        if (name.equals(".git") || name.startsWith(".")) return;

        // Detectar carpetas de librerías, incluyendo 'dependency' y subcarpetas dentro de 'target'
        boolean isLibraryFolder = name.equals("target") 
                || name.equals("lib") 
                || name.equals("libs")
                || name.equals("dependency")
                || name.equals("dependencies")
                || (currentDir.getParentFile() != null && currentDir.getParentFile().getName().equalsIgnoreCase("target"));

        if (isLibraryFolder) {
            File[] jarFiles = currentDir.listFiles((dir, fileName) -> fileName.toLowerCase().endsWith(".jar"));
            if (jarFiles != null) {
                for (File jar : jarFiles) {
                    if (isJarConflictivoConJRE(jar.getName())) {
                        counters[1]++;
                        continue;
                    }
                    IPath jarPath = Path.fromOSString(jar.getAbsolutePath());
                    boolean exists = entries.stream()
                            .filter(e -> e.getEntryKind() == IClasspathEntry.CPE_LIBRARY)
                            .anyMatch(e -> e.getPath().equals(jarPath));
                    if (!exists) {
                        entries.add(JavaCore.newLibraryEntry(jarPath, null, null));
                        counters[0]++;
                    }
                }
            }
        }
        
        File[] children = currentDir.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory()) {
                    findAndAddLibraryJars(child, entries, counters);
                }
            }
        }
    }

    private static void findAndAddSourceFolders(File currentDir, IProject project, List<IClasspathEntry> entries) {
        if (currentDir == null || !currentDir.isDirectory()) return;

        String name = currentDir.getName();
        if (name.equals("target") || name.equals("bin") || name.equals("build") || name.startsWith(".")) {
            return;
        }

        String path = currentDir.getAbsolutePath().replace("\\", "/");
        boolean isMavenSrc = path.endsWith("src/main/java") || path.endsWith("src/test/java");
        boolean isAntSrc = (path.endsWith("/src") || path.endsWith("/test") || path.endsWith("/examples")) && !new File(currentDir, "main/java").exists();

        if (isMavenSrc || isAntSrc) {
            IPath relativePath = new Path(currentDir.getAbsolutePath()).makeRelativeTo(project.getLocation());
            if (relativePath.isEmpty()) {
                entries.add(JavaCore.newSourceEntry(project.getFullPath()));
            } else {
                IFolder folder = project.getFolder(relativePath);
                if (folder.exists()) {
                    entries.add(JavaCore.newSourceEntry(folder.getFullPath()));
                }
            }
            return;
        }

        File[] children = currentDir.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory()) {
                    findAndAddSourceFolders(child, project, entries);
                }
            }
        }
    }
    
    // =====================================================================
    // REPARACIÓN AST DE EMERGENCIA: LOCALIZACIÓN DE VARIABLES
    // =====================================================================
    /**
     * Recorre los nodos AST de los bloques a extraer. Si encuentra un bucle for clásico 
     * inicializado con una variable externa (ej: "i = 0"), la transforma en local 
     * y la renombra (ej: "int i_ext = 0") para evitar colisiones de "Duplicate local variable" 
     * con la declaración original del método.
     */
    private static void localizeVariablesInDocument(ICompilationUnit cu, Document document, List<Position> positions, List<Integer> extIndices) throws Exception {
        ASTParser parser = ASTParser.newParser(AST.JLS25);
        parser.setSource(cu);
        parser.setResolveBindings(true);
        parser.setStatementsRecovery(true);
        parser.setBindingsRecovery(true);
        org.eclipse.jdt.core.dom.CompilationUnit astCu = (org.eclipse.jdt.core.dom.CompilationUnit) parser.createAST(new NullProgressMonitor());
        
        // Activamos el modo de grabación de cambios en el AST
        astCu.recordModifications();
        
        astCu.accept(new org.eclipse.jdt.core.dom.ASTVisitor() {
            @SuppressWarnings("unchecked")
            @Override
            public boolean visit(org.eclipse.jdt.core.dom.ForStatement node) {
                // Comprobar si este bucle for cae dentro del rango de extracción
                boolean inRange = false;
                for (int idx : extIndices) {
                    Position pos = positions.get(idx);
                    if (pos != null && !pos.isDeleted() && 
                        node.getStartPosition() >= pos.getOffset() && 
                        node.getStartPosition() < pos.getOffset() + pos.getLength()) {
                        inRange = true; 
                        break;
                    }
                }
                
                // Transformar si tiene un único inicializador de asignación
                if (inRange && node.initializers().size() == 1) {
                    Object init = node.initializers().get(0);
                    if (init instanceof org.eclipse.jdt.core.dom.Assignment) {
                        org.eclipse.jdt.core.dom.Assignment assignment = (org.eclipse.jdt.core.dom.Assignment) init;
                        
                        if (assignment.getLeftHandSide() instanceof org.eclipse.jdt.core.dom.SimpleName) {
                            AST ast = node.getAST();
                            org.eclipse.jdt.core.dom.SimpleName varName = (org.eclipse.jdt.core.dom.SimpleName) assignment.getLeftHandSide();
                            
                            String oldVarName = varName.getIdentifier();
                            String newVarName = oldVarName + "_ext";
                            
                            // 1. Reemplazar "i = 0" por "int i_ext = 0"
                            org.eclipse.jdt.core.dom.VariableDeclarationFragment fragment = ast.newVariableDeclarationFragment();
                            fragment.setName(ast.newSimpleName(newVarName));
                            fragment.setInitializer((org.eclipse.jdt.core.dom.Expression) org.eclipse.jdt.core.dom.ASTNode.copySubtree(ast, assignment.getRightHandSide()));
                            
                            org.eclipse.jdt.core.dom.VariableDeclarationExpression varDecl = ast.newVariableDeclarationExpression(fragment);
                            varDecl.setType(ast.newPrimitiveType(org.eclipse.jdt.core.dom.PrimitiveType.INT));
                            
                            java.util.List<org.eclipse.jdt.core.dom.ASTNode> initializers = node.initializers();
                            initializers.clear();
                            initializers.add(varDecl);
                            
                            // 2. Visitador para renombrar la variable en el resto del bucle de forma segura
                            org.eclipse.jdt.core.dom.ASTVisitor renamer = new org.eclipse.jdt.core.dom.ASTVisitor() {
                                @Override
                                public boolean visit(org.eclipse.jdt.core.dom.SimpleName name) {
                                    if (name.getIdentifier().equals(oldVarName)) {
                                        org.eclipse.jdt.core.dom.ASTNode parent = name.getParent();
                                        
                                        // Evitar renombrar accidentalmente métodos, campos u objetos (ej: saltarse "obj.i()")
                                        if (parent instanceof org.eclipse.jdt.core.dom.FieldAccess && ((org.eclipse.jdt.core.dom.FieldAccess)parent).getName() == name) return true;
                                        if (parent instanceof org.eclipse.jdt.core.dom.QualifiedName && ((org.eclipse.jdt.core.dom.QualifiedName)parent).getName() == name) return true;
                                        if (parent instanceof org.eclipse.jdt.core.dom.MethodInvocation && ((org.eclipse.jdt.core.dom.MethodInvocation)parent).getName() == name) return true;
                                        if (parent instanceof org.eclipse.jdt.core.dom.SimpleType) return true;
                                        
                                        name.setIdentifier(newVarName);
                                    }
                                    return true;
                                }
                            };
                            
                            // Aplicar el renombramiento a la condición, al incremento y al cuerpo del bucle
                            if (node.getExpression() != null) node.getExpression().accept(renamer);
                            for (Object updater : node.updaters()) {
                                ((org.eclipse.jdt.core.dom.ASTNode)updater).accept(renamer);
                            }
                            node.getBody().accept(renamer);
                        }
                    }
                }
                return true;
            }
        });
        
        org.eclipse.text.edits.TextEdit edit = astCu.rewrite(document, cu.getJavaProject().getOptions(true));
        edit.apply(document);
        cu.getBuffer().setContents(document.get());
    }
}