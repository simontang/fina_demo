package com.fina.b1s.service.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ViewTranslationServiceImplTest {

    @Test
    void translateSqlRewritesLogicalViewAliasWithoutDuplicatingTableAlias() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String sql = "SELECT TOP 50 T0.[DocNum] FROM [VW_ORDR] T0 WHERE T0.[DocDate] < '2026-05-28'";

        String translated = service.translateSql(sql);

        assertEquals(
                "SELECT TOP 50 T0.[DocNum] FROM (SELECT h.[DocEntry], h.[DocNum], h.[DocDate], h.[CardCode], h.[CardName], h.[SlpCode], CAST(NULL AS nvarchar(100)) AS [U_YWLX], l.[LineNum], l.[ItemCode], l.[Dscription], l.[Quantity], l.[WhsCode] AS [WarehouseCode], l.[LineTotal] AS [GTotal], l.[DiscPrcnt], l.[Price], l.[Currency], l.[UomEntry] AS [UoMEntry], l.[UomCode], s.[SlpName], bg.[GroupName], ig.[ItmsGrpNam], wh.[WhsName], CAST(NULL AS nvarchar(100)) AS [Region], CAST(NULL AS nvarchar(100)) AS [Reason] FROM [ORDR] h JOIN [RDR1] l ON h.[DocEntry] = l.[DocEntry] LEFT JOIN [OCRD] bp ON h.[CardCode] = bp.[CardCode] LEFT JOIN [OCRG] bg ON bp.[GroupCode] = bg.[GroupCode] LEFT JOIN [OSLP] s ON h.[SlpCode] = s.[SlpCode] LEFT JOIN [OITM] i ON l.[ItemCode] = i.[ItemCode] LEFT JOIN [OITB] ig ON i.[ItmsGrpCod] = ig.[ItmsGrpCod] LEFT JOIN [OWHS] wh ON l.[WhsCode] = wh.[WhsCode]) AS [T0] WHERE T0.[DocDate] < '2026-05-28'",
                translated
        );
    }

    @Test
    void translateSqlDoesNotTreatGroupByAsAnAlias() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String sql = "SELECT TOP 3 [DocDate] FROM [VW_ORDR] GROUP BY [DocDate]";

        String translated = service.translateSql(sql);

        assertEquals(
                "SELECT TOP 3 [DocDate] FROM (SELECT h.[DocEntry], h.[DocNum], h.[DocDate], h.[CardCode], h.[CardName], h.[SlpCode], CAST(NULL AS nvarchar(100)) AS [U_YWLX], l.[LineNum], l.[ItemCode], l.[Dscription], l.[Quantity], l.[WhsCode] AS [WarehouseCode], l.[LineTotal] AS [GTotal], l.[DiscPrcnt], l.[Price], l.[Currency], l.[UomEntry] AS [UoMEntry], l.[UomCode], s.[SlpName], bg.[GroupName], ig.[ItmsGrpNam], wh.[WhsName], CAST(NULL AS nvarchar(100)) AS [Region], CAST(NULL AS nvarchar(100)) AS [Reason] FROM [ORDR] h JOIN [RDR1] l ON h.[DocEntry] = l.[DocEntry] LEFT JOIN [OCRD] bp ON h.[CardCode] = bp.[CardCode] LEFT JOIN [OCRG] bg ON bp.[GroupCode] = bg.[GroupCode] LEFT JOIN [OSLP] s ON h.[SlpCode] = s.[SlpCode] LEFT JOIN [OITM] i ON l.[ItemCode] = i.[ItemCode] LEFT JOIN [OITB] ig ON i.[ItmsGrpCod] = ig.[ItmsGrpCod] LEFT JOIN [OWHS] wh ON l.[WhsCode] = wh.[WhsCode]) AS [VW_ORDR] GROUP BY [DocDate]",
                translated
        );
    }

    @Test
    void translateSqlRewritesUnquotedLogicalViewNames() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String translated = service.translateSql(
                "SELECT TOP 5 CardCode, SUM(GTotal) FROM VW_OINV GROUP BY CardCode");

        assertFalse(translated.contains("FROM VW_OINV"));
        assertTrue(translated.contains("FROM (SELECT h.[DocEntry]"));
        assertTrue(translated.contains("FROM [OINV] h JOIN [INV1] l"));
    }

    @Test
    void translateSqlBuildsCustomerMasterFieldsFromSapTables() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String translated = service.translateSql(
                "SELECT [CardCode], [PymntGroup], [GroupName], [SlpName] FROM [VW_Customer]");

        assertTrue(translated.contains("FROM [OCRD] bp"));
        assertTrue(translated.contains("LEFT JOIN [OCRG] bpGroup"));
        assertTrue(translated.contains("LEFT JOIN [OCTG] paymentTerms"));
        assertTrue(translated.contains("LEFT JOIN [OSLP] salesperson"));
        assertTrue(translated.contains("paymentTerms.[PymntGroup] AS [PymntGroup]"));
    }

    @Test
    void translateSqlBuildsCustomerBalanceFieldsFromJournalLines() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String translated = service.translateSql(
                "SELECT [ShortName], [BalDue], [DueDate], [Aging], [CreditLimit] FROM [VW_CUSTBAL]");

        assertTrue(translated.contains("FROM [JDT1] journalLine"));
        assertTrue(translated.contains("journalLine.[BalDueDeb] - journalLine.[BalDueCred] AS [BalDue]"));
        assertTrue(translated.contains("journalLine.[DueDate] AS [DueDate]"));
        assertTrue(translated.contains("DATEDIFF(day, journalLine.[DueDate]"));
        assertTrue(translated.contains("bp.[CreditLine] AS [CreditLimit]"));
    }

    @Test
    void translateSqlExposesInvoiceHeaderFields() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String translated = service.translateSql(
                "SELECT [DocTotal], [PaidToDate], [DocDueDate] FROM [VW_OINV]");

        assertTrue(translated.contains("h.[DocTotal]"));
        assertTrue(translated.contains("h.[PaidToDate]"));
        assertTrue(translated.contains("h.[DocDueDate]"));
    }

    @Test
    void translateSqlRewritesObservedSqlServerDialectMismatches() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String sql = """
                SELECT * FROM (
                  SELECT [CardCode] FROM [VW_OINV]
                  ORDER BY [CardCode] LIMIT 10
                ) invoice_rank
                WHERE [DocDate] < CURRENT_DATE
                  AND REGEXP_LIKE([PymntGroup], '([0-9]+)')
                  AND CAST(REGEXP_SUBSTR([PymntGroup], '[0-9]+') AS INTEGER) > 0
                """;

        String translated = service.translateSql(sql);

        assertFalse(translated.toUpperCase().contains("CURRENT_DATE"));
        assertFalse(translated.toUpperCase().contains("REGEXP_LIKE"));
        assertFalse(translated.toUpperCase().contains("REGEXP_SUBSTR"));
        assertFalse(translated.toUpperCase().contains("LIMIT 10"));
        assertTrue(translated.contains("CAST(GETDATE() AS date)"));
        assertTrue(translated.contains("OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY"));
        assertTrue(translated.contains("PATINDEX('%[0-9]%', [PymntGroup])"));
    }

    @Test
    void translateSqlDoesNotRewriteKeywordsInsideStringLiterals() {
        ViewTranslationServiceImpl service = new ViewTranslationServiceImpl();
        service.init();

        String translated = service.translateSql(
                "SELECT 'CURRENT_DATE and LIMIT 10' AS [Label] FROM VW_Customer");

        assertTrue(translated.contains("'CURRENT_DATE and LIMIT 10'"));
        assertTrue(translated.contains("FROM [OCRD] bp"));
    }
}
