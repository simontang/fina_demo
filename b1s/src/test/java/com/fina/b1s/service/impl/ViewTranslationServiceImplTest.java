package com.fina.b1s.service.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
}
