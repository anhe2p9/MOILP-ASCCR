package com.refactoring.extractor.handlers;

import java.io.File;
import java.io.FileReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import com.fasterxml.jackson.core.type.TypeReference;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ResultsProcessor {

    // Estructura para almacenar la información de cada bloque a extraer
	public static class Extraction {
        public int[] range;
        public Integer origIdx;
        public int depth;
        public boolean succeeded = false;
        public String originalMethodName;
        public String failReason = "";

        public Extraction(int[] range, Integer origIdx, int depth, String originalMethodName) {
            this.range = range;
            this.origIdx = origIdx;
            this.depth = depth;
            this.originalMethodName = originalMethodName;
            this.failReason = "";
        }
    }

    private static final Pattern FOLDER_PATTERN = Pattern.compile(
        "^(?<algo>[^_]+)_(?<objs>[^_]+)_(?<classpath>.*\\.java)_(?<method>.+)$"
    );

    private static final ObjectMapper mapper = new ObjectMapper();

    /**
     * Replica exactamente la lógica de 'process_results' en Python:
     * NAVEGA por carpetas, FILTRA por algoritmo, PARSEA el CSV y ORDENA por prioridad.
     */
    public static Map<String, Map<String, List<Extraction>>> processResults(
            String resultsBaseDir, String targetAlgo, List<String> userPriority, String targetClass) {

        Map<String, Map<String, List<Extraction>>> classOffsetsMap = new HashMap<>();
        File baseDir = new File(resultsBaseDir);

        if (!baseDir.exists() || !baseDir.isDirectory()) {
            System.out.println("⚠️ La ruta de resultados no existe o no es un directorio: " + resultsBaseDir);
            return classOffsetsMap;
        }

        List<File> allFolders = new ArrayList<>();
        collectDirectories(baseDir, allFolders);

        for (File folder : allFolders) {
            String folderName = folder.getName();
            Matcher matcher = FOLDER_PATTERN.matcher(folderName);

            if (!matcher.matches()) continue;

            String algo = matcher.group("algo");
            String objsStr = matcher.group("objs");
            String classpath = matcher.group("classpath");
            String method = matcher.group("method");

            if (!algo.equals(targetAlgo)) continue;

            String[] objsArray = objsStr.toLowerCase().split("-");
            if (objsArray.length != 3) continue;

            Set<String> validObjs = new HashSet<>(Arrays.asList("ex", "extractions", "cc", "loc"));
            boolean allValid = true;
            for (String o : objsArray) {
                if (!validObjs.contains(o)) {
                    allValid = false;
                    break;
                }
            }
            if (!allValid) continue;

            if (targetClass != null && !targetClass.isEmpty() && !classpath.endsWith(targetClass)) {
                continue;
            }

            File csvFile = new File(folder, method + "_complete_data.csv");
            if (!csvFile.exists()) continue;

            try {
                CSVRecord bestRow = getBestRowFromCSV(csvFile, objsArray, userPriority);
                if (bestRow == null) continue;

                // Extraer datos de la mejor fila seleccionada
                String offsetsStr = bestRow.get("offsets");
                String infoStr = bestRow.isMapped("solution_info (index,CC,LOC)") ? bestRow.get("solution_info (index,CC,LOC)") : "[]";
                String nestedStr = bestRow.isMapped("nested_solution") ? bestRow.get("nested_solution") : "{}";

                List<int[]> extractedOffsets = parseJsonListIntArray(offsetsStr);
                List<List<Object>> infoList = parseJsonListListObject(infoStr);
                Map<Integer, List<Integer>> nestedMap = parseNestedSolution(nestedStr);

                List<Integer> extIndices = new ArrayList<>();
                if (infoList.size() > 1) {
                    for (int i = 1; i < infoList.size(); i++) {
                        extIndices.add(((Number) infoList.get(i).get(0)).intValue());
                    }
                }

                List<Extraction> methodExtractions = new ArrayList<>();
                String cleanMethod = method.split("-")[0];
                for (int i = 0; i < extractedOffsets.size(); i++) {
                    int[] rng = extractedOffsets.get(i);
                    Integer origIdx = (i < extIndices.size()) ? extIndices.get(i) : null;
                    int depth = (origIdx != null) ? getDepth(origIdx, nestedMap, 0) : 0;

                    methodExtractions.add(new Extraction(rng, origIdx, depth, cleanMethod));
                }

                // Transformar el classpath a la ruta de archivo Java real (reemplazar '-' y '.' por '/')
                String baseCp = classpath.endsWith(".java") ? classpath.substring(0, classpath.length() - 5) : classpath;
	            // Añadimos .replace("-", "/") para corregir el formateo sucio de los CSVs
	            String realClassPath = baseCp.replace(".", "/").replace("-", "/") + ".java";

                classOffsetsMap.putIfAbsent(realClassPath, new HashMap<>());
                classOffsetsMap.get(realClassPath).putIfAbsent(cleanMethod, new ArrayList<>());
                classOffsetsMap.get(realClassPath).get(cleanMethod).addAll(methodExtractions);

            } catch (Exception e) {
                // Silenciamos fallos puntuales de lectura de CSV igual que en Python
            }
        }

        return classOffsetsMap;
    }

    // --- MÉTODOS AUXILIARES ---

    private static void collectDirectories(File current, List<File> result) {
        if (current.isDirectory()) {
            result.add(current);
            File[] files = current.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isDirectory()) collectDirectories(f, result);
                }
            }
        }
    }

    private static CSVRecord getBestRowFromCSV(File csvFile, String[] folderObjs, List<String> userPriority) throws Exception {
        try (Reader reader = new FileReader(csvFile);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build())) {

            List<String> folderObjsList = new ArrayList<>();
            for (String o : folderObjs) {
                folderObjsList.add(o.equals("ex") ? "extractions" : o);
            }

            List<String> usrPrioList = new ArrayList<>();
            for (String p : userPriority) {
                usrPrioList.add(p.equalsIgnoreCase("ex") ? "extractions" : p.toLowerCase());
            }

            List<Integer> sortIndices = new ArrayList<>();
            for (String p : usrPrioList) {
                int idx = folderObjsList.indexOf(p);
                if (idx != -1) sortIndices.add(idx);
            }
            if (sortIndices.isEmpty()) {
                for (int i = 0; i < folderObjsList.size(); i++) sortIndices.add(i);
            }

            CSVRecord bestRecord = null;
            List<Double> bestScores = null;

            for (CSVRecord record : csvParser) {
                String solutionStr = record.get("solution");
                List<Double> parsedSolution = parsePythonListToDouble(solutionStr);

                if (parsedSolution.isEmpty()) continue;

                List<Double> currentScores = new ArrayList<>();
                for (int idx : sortIndices) {
                    if (idx < parsedSolution.size()) {
                        currentScores.add(parsedSolution.get(idx));
                    }
                }

                if (bestRecord == null || compareScores(currentScores, bestScores) < 0) {
                    bestRecord = record;
                    bestScores = currentScores;
                }
            }
            return bestRecord;
        }
    }

    private static int compareScores(List<Double> a, List<Double> b) {
        for (int i = 0; i < Math.min(a.size(), b.size()); i++) {
            int cmp = Double.compare(a.get(i), b.get(i));
            if (cmp != 0) return cmp;
        }
        return 0;
    }

    private static int getDepth(int idx, Map<Integer, List<Integer>> nestedMap, int currentDepth) {
        for (Map.Entry<Integer, List<Integer>> entry : nestedMap.entrySet()) {
            if (entry.getValue() != null && entry.getValue().contains(idx)) {
                return getDepth(entry.getKey(), nestedMap, currentDepth + 1);
            }
        }
        return currentDepth;
    }

    private static List<Double> parsePythonListToDouble(String text) {
        try {
            String json = text.replace("(", "[").replace(")", "]").replace("'", "\"");
            return mapper.readValue(json, new TypeReference<List<Double>>() {});
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private static List<int[]> parseJsonListIntArray(String text) {
        try {
            String json = text.replace("'", "\"");
            return mapper.readValue(json, new TypeReference<List<int[]>>() {});
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private static List<List<Object>> parseJsonListListObject(String text) {
        try {
            String json = text.replace("(", "[").replace(")", "]").replace("'", "\"");
            return mapper.readValue(json, new TypeReference<List<List<Object>>>() {});
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private static Map<Integer, List<Integer>> parseNestedSolution(String text) {
        try {
            String json = text.replace("'", "\"");
            return mapper.readValue(json, new TypeReference<Map<Integer, List<Integer>>>() {});
        } catch (Exception e) {
            return Collections.emptyMap();
        }
    }
}