package com.refactoring.extractor.handlers;

import java.io.File;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class PomPatcher {

    // Método genérico: le pasas el artifact que falló, el nuevo groupId (si cambia) y la nueva versión
    public static void autoPatchDependency(File pomFile, String targetArtifactId, String newGroupId, String newVersion) {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(pomFile);
            doc.getDocumentElement().normalize();

            // Buscar todas las dependencias
            NodeList dependencies = doc.getElementsByTagName("dependency");

            boolean changed = false;

            for (int i = 0; i < dependencies.getLength(); i++) {
                Element dep = (Element) dependencies.item(i);
                String artifactId = getTagValue("artifactId", dep);

                // Si encontramos la dependencia que causó el error de Maven
                if (artifactId != null && artifactId.equals(targetArtifactId)) {
                    
                    // 1. Cambiamos la versión dinámicamente
                    updateTagValue("version", dep, newVersion, doc);
                    
                    // 2. Si hay un nuevo groupId, lo cambiamos
                    if (newGroupId != null && !newGroupId.isEmpty()) {
                        updateTagValue("groupId", dep, newGroupId, doc);
                    }
                    changed = true;
                }
            }

            // Si hicimos cambios, guardamos el pom.xml sobreescribiéndolo
            if (changed) {
                TransformerFactory transformerFactory = TransformerFactory.newInstance();
                Transformer transformer = transformerFactory.newTransformer();
                DOMSource source = new DOMSource(doc);
                StreamResult result = new StreamResult(pomFile);
                transformer.transform(source, result);
                System.out.println("🔧 pom.xml parcheado automáticamente para: " + targetArtifactId);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String getTagValue(String tag, Element element) {
        NodeList nodeList = element.getElementsByTagName(tag);
        if (nodeList != null && nodeList.getLength() > 0) {
            return nodeList.item(0).getTextContent();
        }
        return null;
    }

    private static void updateTagValue(String tag, Element parent, String newValue, Document doc) {
        NodeList nodeList = parent.getElementsByTagName(tag);
        if (nodeList != null && nodeList.getLength() > 0) {
            nodeList.item(0).setTextContent(newValue);
        } else {
            // Si la etiqueta no existe (ej. no tenía <version>), la creamos
            Element newElement = doc.createElement(tag);
            newElement.setTextContent(newValue);
            parent.appendChild(newElement);
        }
    }
}