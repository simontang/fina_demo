package com.fina.b1s.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.service.ViewTranslationService;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
public class ViewTranslationServiceImpl implements ViewTranslationService {

    private static final String TRANSLATION_PATH = "meta/view-translations.json";
    private static final Pattern QUOTED_IDENTIFIER = Pattern.compile("\"([A-Za-z_][A-Za-z0-9_]*)\"");
    private static final Pattern TO_NVARCHAR = Pattern.compile(
            "TO_NVARCHAR\\(\\[([^]]+)]\\s*,\\s*'([^']+)'\\)");

    private final ObjectMapper mapper = new ObjectMapper();
    private final Map<String, String> translations = new LinkedHashMap<>();

    @PostConstruct
    public void init() {
        loadConfiguredTranslations();
        loadFallbackTranslations();
        log.info("Loaded {} logical view translations", translations.size());
    }

    @Override
    public String translateSql(String sql) {
        if (!StringUtils.hasText(sql)) {
            return sql;
        }
        String translated = QUOTED_IDENTIFIER.matcher(sql).replaceAll("[$1]");
        translated = translateDateFormats(translated);
        translated = replaceLogicalViews(translated);
        translated = replaceTrailingLimit(translated);
        return translated;
    }

    private String replaceLogicalViews(String sql) {
        String translated = sql;
        for (Map.Entry<String, String> entry : translations.entrySet()) {
            Pattern pattern = Pattern.compile("(?i)\\b(FROM|JOIN)\\s+\\[" + Pattern.quote(entry.getKey()) + "\\]");
            Matcher matcher = pattern.matcher(translated);
            StringBuffer out = new StringBuffer();
            while (matcher.find()) {
                String clause = matcher.group(1);
                matcher.appendReplacement(out, Matcher.quoteReplacement(clause + " " + entry.getValue()));
            }
            matcher.appendTail(out);
            translated = out.toString();
        }
        return translated;
    }

    private void loadConfiguredTranslations() {
        ClassPathResource resource = new ClassPathResource(TRANSLATION_PATH);
        if (!resource.exists()) {
            return;
        }
        try (InputStream is = resource.getInputStream()) {
            JsonNode root = mapper.readTree(is);
            root.fields().forEachRemaining(entry -> translations.put(entry.getKey(), entry.getValue().asText()));
        } catch (Exception e) {
            log.warn("Failed to load {}: {}", TRANSLATION_PATH, e.getMessage());
        }
    }

    private void loadFallbackTranslations() {
        translations.putIfAbsent("VW_ORDR", lineView("VW_ORDR", "ORDR", "RDR1"));
        translations.putIfAbsent("VW_ODLN", lineView("VW_ODLN", "ODLN", "DLN1"));
        translations.putIfAbsent("VW_OINV", lineView("VW_OINV", "OINV", "INV1"));
        translations.putIfAbsent("VW_ORDN", lineView("VW_ORDN", "ORDN", "RDN1"));
    }

    private static String lineView(String alias, String headerTable, String lineTable) {
        return "(SELECT h.[DocEntry], h.[DocNum], h.[DocDate], h.[CardCode], h.[CardName], h.[SlpCode], "
                + "CAST(NULL AS nvarchar(100)) AS [U_YWLX], l.[LineNum], l.[ItemCode], l.[Dscription], l.[Quantity], "
                + "l.[WhsCode] AS [WarehouseCode], l.[LineTotal] AS [GTotal], l.[DiscPrcnt], l.[Price], "
                + "l.[Currency], l.[UomEntry] AS [UoMEntry], l.[UomCode], "
                + "s.[SlpName], bg.[GroupName], ig.[ItmsGrpNam], wh.[WhsName], "
                + "CAST(NULL AS nvarchar(100)) AS [Region], CAST(NULL AS nvarchar(100)) AS [Reason] "
                + "FROM [" + headerTable + "] h "
                + "JOIN [" + lineTable + "] l ON h.[DocEntry] = l.[DocEntry] "
                + "LEFT JOIN [OCRD] bp ON h.[CardCode] = bp.[CardCode] "
                + "LEFT JOIN [OCRG] bg ON bp.[GroupCode] = bg.[GroupCode] "
                + "LEFT JOIN [OSLP] s ON h.[SlpCode] = s.[SlpCode] "
                + "LEFT JOIN [OITM] i ON l.[ItemCode] = i.[ItemCode] "
                + "LEFT JOIN [OITB] ig ON i.[ItmsGrpCod] = ig.[ItmsGrpCod] "
                + "LEFT JOIN [OWHS] wh ON l.[WhsCode] = wh.[WhsCode]) AS [" + alias + "]";
    }

    private static String translateDateFormats(String sql) {
        Matcher matcher = TO_NVARCHAR.matcher(sql);
        StringBuffer out = new StringBuffer();
        while (matcher.find()) {
            String field = matcher.group(1);
            String hanaFormat = matcher.group(2);
            String replacement = switch (hanaFormat) {
                case "YYYY" -> "CONVERT(varchar(4), [" + field + "], 120)";
                case "YYYY-MM" -> "CONVERT(varchar(7), [" + field + "], 120)";
                case "YYYY-MM-DD" -> "CONVERT(varchar(10), [" + field + "], 120)";
                case "YYYY-IW" -> "CONCAT(YEAR([" + field + "]), '-', DATEPART(ISO_WEEK, [" + field + "]))";
                default -> "[" + field + "]";
            };
            matcher.appendReplacement(out, Matcher.quoteReplacement(replacement));
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static String replaceTrailingLimit(String sql) {
        Matcher matcher = Pattern.compile("(?is)\\s+LIMIT\\s+(\\d+)\\s*$").matcher(sql);
        if (!matcher.find()) {
            return sql;
        }
        String limit = matcher.group(1);
        String withoutLimit = matcher.replaceFirst("");
        return withoutLimit.replaceFirst("(?is)^\\s*SELECT\\s+", "SELECT TOP " + limit + " ");
    }
}
