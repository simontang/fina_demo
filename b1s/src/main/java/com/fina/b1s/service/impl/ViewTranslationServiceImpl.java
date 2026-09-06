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
    private static final String FIELD_REFERENCE =
            "(?:\\[[A-Za-z_][A-Za-z0-9_]*]|[A-Za-z_][A-Za-z0-9_]*)"
                    + "(?:\\s*\\.\\s*(?:\\[[A-Za-z_][A-Za-z0-9_]*]|[A-Za-z_][A-Za-z0-9_]*))?";
    private static final Pattern REGEXP_SUBSTR = Pattern.compile(
            "(?i)REGEXP_SUBSTR\\(\\s*(" + FIELD_REFERENCE + ")\\s*,\\s*'([^']*)'\\s*\\)");
    private static final Pattern REGEXP_LIKE = Pattern.compile(
            "(?i)REGEXP_LIKE\\(\\s*(" + FIELD_REFERENCE + ")\\s*,\\s*'([^']*)'\\s*\\)");
    private static final Pattern CURRENT_DATE = Pattern.compile("(?i)\\bCURRENT_DATE\\b");
    private static final Pattern REMAINING_LIMIT = Pattern.compile("(?i)\\bLIMIT\\s+(\\d+)\\b");

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
        translated = translateSqlServerFunctions(translated);
        translated = replaceLogicalViews(translated);
        translated = replaceLimits(translated);
        return translated;
    }

    private String replaceLogicalViews(String sql) {
        String translated = sql;
        for (Map.Entry<String, String> entry : translations.entrySet()) {
            String logicalName = Pattern.quote(entry.getKey());
            Pattern pattern = Pattern.compile("(?i)\\b(FROM|JOIN)\\s+(?:\\[" + logicalName + "\\]|"
                    + logicalName + "(?![A-Za-z0-9_]))(?:\\s+(?:AS\\s+)?"
                    + "(?!(?:WHERE|GROUP|ORDER|HAVING|LIMIT|JOIN|LEFT|RIGHT|INNER|FULL|CROSS|ON|UNION)\\b)"
                    + "([A-Za-z_][A-Za-z0-9_]*|\\[[A-Za-z_][A-Za-z0-9_]*]))?");
            Matcher matcher = pattern.matcher(translated);
            StringBuffer out = new StringBuffer();
            while (matcher.find()) {
                String clause = matcher.group(1);
                String alias = matcher.group(2);
                String replacement = entry.getValue();
                if (StringUtils.hasText(alias)) {
                    replacement = replaceTranslationAlias(replacement, normalizeAlias(alias));
                }
                matcher.appendReplacement(out, Matcher.quoteReplacement(clause + " " + replacement));
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
        translations.putIfAbsent("VW_ORDN", lineView("VW_ORDN", "ORDN", "RDN1"));

        // These logical views need SQL Server-specific joins rather than direct table aliases.
        translations.putIfAbsent("VW_OINV", invoiceLineView());
        translations.putIfAbsent("VW_CUSTBAL", balanceView("VW_CUSTBAL", "C"));
        translations.putIfAbsent("VW_DEALBAL", balanceView("VW_DEALBAL", "S"));
        translations.putIfAbsent("VW_Customer", businessPartnerView("VW_Customer", "C"));
        translations.putIfAbsent("VW_Supplier", businessPartnerView("VW_Supplier", "S"));
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

    private static String invoiceLineView() {
        return "(SELECT h.[DocEntry], h.[DocNum], h.[DocDate], h.[DocDueDate], h.[CardCode], h.[CardName], "
                + "h.[DocCur] AS [DocCurrency], h.[DocTotal], h.[PaidToDate], h.[DocStatus], h.[CANCELED], "
                + "h.[SlpCode], CAST(NULL AS nvarchar(100)) AS [U_YWLX], l.[LineNum], l.[ItemCode], "
                + "l.[Dscription], l.[Quantity], l.[WhsCode] AS [WarehouseCode], l.[LineTotal] AS [GTotal], "
                + "l.[DiscPrcnt], l.[Price], l.[Currency], l.[UomEntry] AS [UoMEntry], l.[UomCode], "
                + "s.[SlpName], bg.[GroupName], ig.[ItmsGrpNam], wh.[WhsName], "
                + "CAST(NULL AS nvarchar(100)) AS [Region], CAST(NULL AS nvarchar(100)) AS [Reason] "
                + "FROM [OINV] h JOIN [INV1] l ON h.[DocEntry] = l.[DocEntry] "
                + "LEFT JOIN [OCRD] bp ON h.[CardCode] = bp.[CardCode] "
                + "LEFT JOIN [OCRG] bg ON bp.[GroupCode] = bg.[GroupCode] "
                + "LEFT JOIN [OSLP] s ON h.[SlpCode] = s.[SlpCode] "
                + "LEFT JOIN [OITM] i ON l.[ItemCode] = i.[ItemCode] "
                + "LEFT JOIN [OITB] ig ON i.[ItmsGrpCod] = ig.[ItmsGrpCod] "
                + "LEFT JOIN [OWHS] wh ON l.[WhsCode] = wh.[WhsCode]) AS [VW_OINV]";
    }

    private static String businessPartnerView(String alias, String cardType) {
        return "(SELECT bp.[CardCode], bp.[CardName], bp.[CardType], bp.[Currency], bp.[LicTradNum], "
                + "bp.[validFor], bpGroup.[GroupName], salesperson.[SlpName], bp.[State1] AS [Province], "
                + "CAST(NULL AS nvarchar(100)) AS [Region], bp.[GTSRegNum], bp.[GTSBankAct], bp.[GTSBilAddr], "
                + "bp.[ECVatGroup], CAST(NULL AS nvarchar(100)) AS [VatName], bp.[Phone1], bp.[Phone2], "
                + "bp.[E_Mail], bp.[CntctPrsn], contact.[Position], contact.[Address] AS [CntAddress], "
                + "contact.[Tel1], contact.[Cellolar], contact.[E_MailL], bp.[Address], bp.[MailAddres], "
                + "paymentTerms.[PymntGroup] AS [PymntGroup], "
                + "COALESCE(paymentTerms.[ExtraDays], 0) + COALESCE(paymentTerms.[ExtraMonth], 0) * 30 AS [PaymentTermDays], "
                + "priceList.[ListName], bp.[BankCountr], bp.[BankCode], bank.[BankName], bp.[DflAccount], "
                + "bp.[DflSwift], bp.[DflBranch], bp.[UseShpdGd], bp.[ConnBP], connectedBp.[CardName] AS [ConnBPName], "
                + "bp.[FatherType], bp.[FatherCard], parentBp.[CardName] AS [FatherName], bp.[GroupNum], "
                + "bp.[Balance], bp.[CreditLine], bp.[CreateDate], bp.[UpdateDate] "
                + "FROM [OCRD] bp "
                + "LEFT JOIN [OCRG] bpGroup ON bp.[GroupCode] = bpGroup.[GroupCode] "
                + "LEFT JOIN [OSLP] salesperson ON bp.[SlpCode] = salesperson.[SlpCode] "
                + "LEFT JOIN [OCTG] paymentTerms ON bp.[GroupNum] = paymentTerms.[GroupNum] "
                + "LEFT JOIN [OPLN] priceList ON bp.[ListNum] = priceList.[ListNum] "
                + "LEFT JOIN [ODSC] bank ON bp.[BankCode] = bank.[BankCode] "
                + "LEFT JOIN [OCPR] contact ON bp.[CardCode] = contact.[CardCode] AND bp.[CntctPrsn] = contact.[Name] "
                + "LEFT JOIN [OCRD] connectedBp ON bp.[ConnBP] = connectedBp.[CardCode] "
                + "LEFT JOIN [OCRD] parentBp ON bp.[FatherCard] = parentBp.[CardCode] "
                + "WHERE bp.[CardType] = '" + cardType + "') AS [" + alias + "]";
    }

    private static String balanceView(String alias, String cardType) {
        return "(SELECT journalLine.[ShortName], bp.[CardCode], bp.[CardName], bp.[State1] AS [Province], "
                + "CAST(NULL AS nvarchar(100)) AS [Region], bp.[CardType], bpGroup.[GroupName], "
                + "salesperson.[SlpName], journalLine.[RefDate] AS [RefDate], "
                + "journalLine.[DueDate] AS [DueDate], journalLine.[TransId] AS [TransId], "
                + "COALESCE(NULLIF(journalLine.[FCCurrency], ''), bp.[Currency]) AS [Currency], "
                + "CASE WHEN NULLIF(LTRIM(RTRIM(journalLine.[FCCurrency])), '') IS NULL THEN 'Y' ELSE 'N' END AS [IsLocCurrency], "
                + "journalLine.[Debit] - journalLine.[Credit] AS [Amount], "
                + "journalLine.[BalDueDeb] - journalLine.[BalDueCred] AS [BalDue], "
                + "journalLine.[FCDebit] - journalLine.[FCCredit] AS [FCAmount], "
                + "journalLine.[BalFcDeb] - journalLine.[BalFcCred] AS [BalFcDue], "
                + "CONVERT(varchar(20), DATEDIFF(day, journalLine.[DueDate], CAST(GETDATE() AS date))) AS [Aging], "
                + "DATEDIFF(day, journalLine.[DueDate], CAST(GETDATE() AS date)) AS [AgingDays], "
                + "journalLine.[BPLId], journalLine.[BPLName], bp.[Balance], bp.[CreditLine], "
                + "bp.[CreditLine] AS [CreditLimit], bp.[GroupNum], paymentTerms.[PymntGroup], "
                + "COALESCE(paymentTerms.[ExtraDays], 0) + COALESCE(paymentTerms.[ExtraMonth], 0) * 30 AS [PaymentTermDays] "
                + "FROM [JDT1] journalLine "
                + "JOIN [OCRD] bp ON journalLine.[ShortName] = bp.[CardCode] AND bp.[CardType] = '" + cardType + "' "
                + "LEFT JOIN [OCRG] bpGroup ON bp.[GroupCode] = bpGroup.[GroupCode] "
                + "LEFT JOIN [OSLP] salesperson ON bp.[SlpCode] = salesperson.[SlpCode] "
                + "LEFT JOIN [OCTG] paymentTerms ON bp.[GroupNum] = paymentTerms.[GroupNum] "
                + "WHERE journalLine.[BalDueDeb] <> 0 OR journalLine.[BalDueCred] <> 0 "
                + "OR journalLine.[BalFcDeb] <> 0 OR journalLine.[BalFcCred] <> 0) AS [" + alias + "]";
    }

    private static String normalizeAlias(String alias) {
        if (alias.startsWith("[") && alias.endsWith("]")) {
            return alias.substring(1, alias.length() - 1);
        }
        return alias;
    }

    private static String replaceTranslationAlias(String translation, String alias) {
        return translation.replaceFirst("(?is)\\)\\s+AS\\s+\\[[A-Za-z_][A-Za-z0-9_]*]\\s*$", ") AS [" + alias + "]");
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

    private static String translateSqlServerFunctions(String sql) {
        String translated = replaceRegexpSubstr(sql);
        translated = replaceRegexpLike(translated);
        return replaceCurrentDate(translated);
    }

    private static String replaceCurrentDate(String sql) {
        Matcher matcher = CURRENT_DATE.matcher(sql);
        StringBuffer out = new StringBuffer();
        while (matcher.find()) {
            if (isInsideStringLiteral(sql, matcher.start())) {
                continue;
            }
            matcher.appendReplacement(out, "CAST(GETDATE() AS date)");
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static String replaceRegexpSubstr(String sql) {
        Matcher matcher = REGEXP_SUBSTR.matcher(sql);
        StringBuffer out = new StringBuffer();
        while (matcher.find()) {
            String field = matcher.group(1);
            String pattern = matcher.group(2);
            if (!isDigitPattern(pattern)) {
                continue;
            }
            matcher.appendReplacement(out, Matcher.quoteReplacement(firstDigitSequence(field)));
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static String replaceRegexpLike(String sql) {
        Matcher matcher = REGEXP_LIKE.matcher(sql);
        StringBuffer out = new StringBuffer();
        while (matcher.find()) {
            String field = matcher.group(1);
            String pattern = matcher.group(2);
            if (!isDigitPattern(pattern)) {
                continue;
            }
            String replacement = "^[0-9]+$".equals(stripCapturingParentheses(pattern))
                    ? "TRY_CONVERT(bigint, " + field + ") IS NOT NULL"
                    : "PATINDEX('%[0-9]%', " + field + ") > 0";
            matcher.appendReplacement(out, Matcher.quoteReplacement(replacement));
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static boolean isDigitPattern(String pattern) {
        return stripCapturingParentheses(pattern).contains("[0-9]+");
    }

    private static String stripCapturingParentheses(String pattern) {
        return pattern.replace("(", "").replace(")", "");
    }

    private static String firstDigitSequence(String field) {
        String firstDigit = "PATINDEX('%[0-9]%', " + field + ")";
        return "CASE WHEN " + firstDigit + " = 0 THEN NULL ELSE SUBSTRING(" + field + ", " + firstDigit
                + ", PATINDEX('%[^0-9]%', SUBSTRING(" + field + ", " + firstDigit
                + ", 8000) + 'X') - 1) END";
    }

    private static String replaceLimits(String sql) {
        String translated = replaceTrailingLimit(sql);
        Matcher matcher = REMAINING_LIMIT.matcher(translated);
        StringBuffer out = new StringBuffer();
        while (matcher.find()) {
            if (isInsideStringLiteral(translated, matcher.start())) {
                continue;
            }
            matcher.appendReplacement(out, "OFFSET 0 ROWS FETCH NEXT " + matcher.group(1) + " ROWS ONLY");
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static boolean isInsideStringLiteral(String sql, int position) {
        boolean inside = false;
        for (int i = 0; i < position; i++) {
            if (sql.charAt(i) != '\'') {
                continue;
            }
            if (inside && i + 1 < position && sql.charAt(i + 1) == '\'') {
                i++;
                continue;
            }
            inside = !inside;
        }
        return inside;
    }

    private static String replaceTrailingLimit(String sql) {
        Matcher matcher = Pattern.compile("(?is)\\s+LIMIT\\s+(\\d+)\\s*;?\\s*$").matcher(sql);
        if (!matcher.find()) {
            return sql;
        }
        String limit = matcher.group(1);
        String withoutLimit = matcher.replaceFirst("");
        return withoutLimit.replaceFirst("(?is)^\\s*SELECT\\s+", "SELECT TOP " + limit + " ");
    }
}
