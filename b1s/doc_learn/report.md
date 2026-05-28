# Mashina Corporation PO: PDF Extraction and B1 Validation

## Executive Summary

The source PDF is a valid purchase order, but it is not ready to post into SAP B1 as-is.
The B1 master data for `C42000` points to a US-based customer in USD, with contact `Anthony Smith` as the default contact and payment terms group `1`, which resolves to `2P10Net30`.
The PDF still uses a China ship-to address, `Luna Hu`, `IDR`, and a literal `1 Days` payment term string, so those fields should be rewritten before any B1 submission.

The safest revision is to keep the original 3 printer lines, switch the header to the BP's US address/contact, and use B1 recent transaction prices for the original lines.
If you want a 10-line test PO, the extra 7 lines should be treated as a template expansion, not as literal extraction from the PDF.

## 1. PDF Extraction

Source file:

`/Users/cid/Downloads/Mashina_Corporation_Purchase_Order_revised.pdf`

Extracted fields:

| Field | Value |
| --- | --- |
| Company | Mashina Corporation |
| Ship to | No. 12, Longhua Road, Shenzhen, China |
| Ship date | Oct. 31, 2025 |
| Payment terms | 1 Days |
| Contact person | Luna Hu |
| PO number | NO. LF-JL-250077 |
| Date | Oct. 2, 2025 |
| Currency | IDR |

Extracted item lines:

| Line | Description | Qty | Unit | VAT |
| --- | --- | --- | --- | --- |
| 1 | J.B. Officeprint 1420 | 5 | pcs | 11 |
| 2 | J.B. Officeprint 1111 | 5 | pcs | 11 |
| 3 | J.B. Officeprint 1186 | 10 | pcs | 11 |

## 2. B1 Service Layer Evidence

Queried through `https://b1s.alphafina.cn/b1s/v1` with:

- `X-Tenant-Id: 2`
- `X-Company-DB: SBODemoUS`

### Business Partner

`BusinessPartners('C42000')`

| Field | Value |
| --- | --- |
| CardCode | C42000 |
| CardName | Mashina Corporation |
| Currency | `$` |
| PriceListNum | 4 |
| PayTermsGrpCode | 1 |
| Default contact | Anthony Smith |

Addresses:

| Type | Address |
| --- | --- |
| Bill To | 406 Ingram Ave, Dayton, OH 45417, US |
| Ship To | 406 Ingram Ave, Dayton, OH 45417, US |

Contacts:

| InternalCode | Name | Email | Position |
| --- | --- | --- | --- |
| 9 | Anthony Smith | anthony.smith@mashina.sap.com | General Manager |
| 10 | Jennifer Herren | jennifer.herren@mashina.sap.com | Sales Manager |

### Payment Terms

`PaymentTermsTypes(1)`

| Field | Value |
| --- | --- |
| GroupNumber | 1 |
| PaymentTermsGroupName | 2P10Net30 |
| NumberOfAdditionalDays | 30 |
| PriceListNo | 1 |

### Recent Sales Order

`Orders(1184)`

| Field | Value |
| --- | --- |
| CardCode | C42000 |
| ContactPersonCode | 9 |
| PaymentGroupCode | 1 |
| ShipToCode | Ship To |
| PayToCode | Bill To |
| DocCurrency | `$` |
| DocDate | 2026-05-23 |
| DocDueDate | 2026-05-30 |
| TaxDate | 2026-05-23 |
| ShipDate | 2026-05-30 for every line |

Recent line prices:

| ItemCode | ItemDescription | UnitPrice |
| --- | --- | --- |
| A00001 | J.B. Officeprint 1420 | 100 |
| A00002 | J.B. Officeprint 1111 | 200 |
| A00003 | J.B. Officeprint 1186 | 300 |

### Price List 4 vs Recent Transaction

BP `PriceListNum` is 4, so list 4 is the default pricing reference for the customer.
For the original three items, price list 4 is materially higher than the recent transaction prices:

| ItemCode | Price list 4 | Recent order price | Comment |
| --- | --- | --- | --- |
| A00001 | 500 | 100 | +400% vs recent order |
| A00002 | 250 | 200 | +25% vs recent order |
| A00003 | 375 | 300 | +25% vs recent order |

This is the main pricing warning in the feedback: do not blindly trust the price list when the latest transaction is materially lower.

## 3. What Must Change

| PDF field | B1 reference | Action |
| --- | --- | --- |
| Ship date | 2025-10-31 | Move to a future date; use `2026-05-30` as the working B1 reference |
| Contact person | Luna Hu | Replace with `Anthony Smith` (contact code `9`) |
| Ship-to address | Shenzhen, China | Remove CN address and use the US Ship To / Bill To address from BP master |
| Currency | IDR | Use `$` / USD; no FX conversion should be applied |
| Payment terms | `1 Days` | Replace with B1 group `1` = `2P10Net30` |
| Unit prices | Missing in PDF | Use recent transaction prices for the original 3 lines |
| Price reference | N/A | Price list 4 is too high for the original 3 lines; use recent price as the better reference |
| Store code | Missing | Leave blank unless a real store mapping is provided |

## 4. Recommended 10-Line Revision

The first 3 lines are from the PDF and should stay in place.
The extra 7 lines below are a template expansion for testing and should be treated as B1-valid examples, not literal extraction from the PDF.

| Line | ItemCode | Description | Qty | UnitPrice | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | A00001 | J.B. Officeprint 1420 | 5 | 100 | recent order 1184 |
| 2 | A00002 | J.B. Officeprint 1111 | 5 | 200 | recent order 1184 |
| 3 | A00003 | J.B. Officeprint 1186 | 10 | 300 | recent order 1184 |
| 4 | A00004 | Rainbow Color Printer 5.0 | 1 | 625 | price list 4 |
| 5 | A00005 | Rainbow Color Printer 7.5 | 1 | 500 | price list 4 |
| 6 | A00006 | Rainbow 1200 Laser Series | 1 | 500 | price list 4 |
| 7 | B10000 | Printer Label | 100 | 1.25 | price list 4 |
| 8 | I00007 | Rainbow Printer 9.5 Inkjet Cartridge | 10 | 35 | price list 4 |
| 9 | LM4029 | LeMon 4029 Printer | 2 | 300 | price list 4 |
| 10 | R00001 | Printer Paper A4 White | 50 | 6.25 | price list 4 |

All of these item codes exist in `SBODemoUS`, and each has a valid `WarehouseCode = 02` row in the item warehouse info.
Some of the added items have zero stock, so warehouse feasibility still needs a separate check.

## 5. Suggested B1 Payload

If you want to post a live order, use `/b1s/v1/Orders`.
If you want a draft, use `/b1s/v1/Drafts` and add `DocObjectCode: oOrders`.

```json
{
  "CardCode": "C42000",
  "DocDate": "2026-05-23",
  "DocDueDate": "2026-05-30",
  "TaxDate": "2026-05-23",
  "ContactPersonCode": 9,
  "PaymentGroupCode": 1,
  "ShipToCode": "Ship To",
  "PayToCode": "Bill To",
  "DocCurrency": "$",
  "Comments": "Revised from sample PDF after B1 validation",
  "DocumentLines": [
    { "ItemCode": "A00001", "Quantity": 5, "WarehouseCode": "02", "UnitPrice": 100, "ShipDate": "2026-05-30" },
    { "ItemCode": "A00002", "Quantity": 5, "WarehouseCode": "02", "UnitPrice": 200, "ShipDate": "2026-05-30" },
    { "ItemCode": "A00003", "Quantity": 10, "WarehouseCode": "02", "UnitPrice": 300, "ShipDate": "2026-05-30" },
    { "ItemCode": "A00004", "Quantity": 1, "WarehouseCode": "02", "UnitPrice": 625, "ShipDate": "2026-05-30" },
    { "ItemCode": "A00005", "Quantity": 1, "WarehouseCode": "02", "UnitPrice": 500, "ShipDate": "2026-05-30" },
    { "ItemCode": "A00006", "Quantity": 1, "WarehouseCode": "02", "UnitPrice": 500, "ShipDate": "2026-05-30" },
    { "ItemCode": "B10000", "Quantity": 100, "WarehouseCode": "02", "UnitPrice": 1.25, "ShipDate": "2026-05-30" },
    { "ItemCode": "I00007", "Quantity": 10, "WarehouseCode": "02", "UnitPrice": 35, "ShipDate": "2026-05-30" },
    { "ItemCode": "LM4029", "Quantity": 2, "WarehouseCode": "02", "UnitPrice": 300, "ShipDate": "2026-05-30" },
    { "ItemCode": "R00001", "Quantity": 50, "WarehouseCode": "02", "UnitPrice": 6.25, "ShipDate": "2026-05-30" }
  ]
}
```

## 6. Bottom Line

- Keep the original 3 lines, but rewrite the header to match B1 master data.
- Use `Anthony Smith`, not `Luna Hu`.
- Use the US address and `$` currency.
- Treat payment terms as `2P10Net30`, not the literal `1 Days`.
- Use recent transaction prices for the original 3 items; price list 4 is too high for that set.
- Only the extra 7 lines are a controlled expansion for testing the B1 path.
