// Run only against a disposable local PostgreSQL database and local Metrics Server.
// METRICS_BASE_URL=http://127.0.0.1:15704 PGPORT=55486 PGDATABASE=scope_smoke PGUSER=scope_test node scripts/test-visible-scope.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

const base = process.env.METRICS_BASE_URL || 'http://127.0.0.1:15704';
assert.equal(new URL(base).hostname, '127.0.0.1', 'This smoke test only supports disposable localhost services');
const port = process.env.PGPORT || '55486';
const database = process.env.PGDATABASE || 'scope_smoke';
const username = process.env.PGUSER || 'scope_test';
let checks = 0;
function sql(statement) {
  return execFileSync(process.env.PSQL_BIN || 'psql', [
    '-X', '-h', '127.0.0.1', '-p', port, '-U', username, '-d', database,
    '-v', 'ON_ERROR_STOP=1', '-At', '-c', statement,
  ], { encoding: 'utf8' }).trim();
}
async function api(method, path, body, expected = 200, tenant) {
  const headers = { 'Content-Type': 'application/json' };
  if (tenant) headers['X-Tenant-Id'] = tenant;
  const response = await fetch(`${base}/api/v1${path}`, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json();
  assert.equal(response.status, expected, `${method} ${path}: ${JSON.stringify(payload)}`);
  checks += 1;
  return payload.data;
}

sql(`CREATE TABLE IF NOT EXISTS public.hankel_scope_smoke_sales (region TEXT, amount NUMERIC, quantity INT);
     TRUNCATE public.hankel_scope_smoke_sales;
     INSERT INTO public.hankel_scope_smoke_sales VALUES ('East', 100, 2), ('East', 50, 1), ('West', 75, 3);
     CREATE TABLE IF NOT EXISTS public.hidden_scope_smoke_sales (secret TEXT);
     CREATE SCHEMA IF NOT EXISTS scope_hidden;
     CREATE TABLE IF NOT EXISTS scope_hidden.hankel_scope_smoke_sales (secret TEXT);`);
const config = { name: 'Hankel Visible Scope Local Smoke', url: `jdbc:postgresql://127.0.0.1:${port}/${database}`,
  username, password: 'local-smoke', schemaName: 'public', sourceType: 'cdp_postgres', status: 1 };
const ds = await api('POST', '/datasources', config);
const dsPath = `/datasources/${ds.id}`;
assert.equal(ds.visibleScopeMode, 'RESTRICTED');
await api('POST', `${dsPath}/query`, { sql: 'SELECT 1' }, 403);
assert.deepEqual(await api('GET', `${dsPath}/schema/tables`), []);

const rule = { schemaName: 'public', tablePattern: 'hankel_', patternType: 'PREFIX', caseSensitive: false, status: 1 };
const scope = await api('POST', `${dsPath}/visible-scopes`, rule, 200, 'unrelated-agent-tenant');
assert.equal(Object.hasOwn(scope, 'tenantId'), false);
const defaultScopes = await api('GET', `${dsPath}/visible-scopes`);
assert.deepEqual(await api('GET', `${dsPath}/visible-scopes`, undefined, 200, 'tenant_5'), defaultScopes);
const legacy = await api('GET', `${dsPath}/table-grants`, undefined, 200, 'hankel');
assert.equal(legacy[0].tenantId, '__datasource__');
await api('PUT', `${dsPath}/visible-scopes/${scope.id}`, { ...rule, tablePattern: 'hankel_scope_smoke_sales', patternType: 'EXACT' });
await api('PUT', `${dsPath}/table-grants/${scope.id}`, rule);
const extraScope = await api('POST', `${dsPath}/table-grants`, { ...rule, tablePattern: 'hankel_unused', patternType: 'EXACT' });
await api('DELETE', `${dsPath}/visible-scopes/${extraScope.id}`);
const tables = await api('GET', `${dsPath}/schema/tables`);
assert.ok(tables.some(t => t.tableName === 'hankel_scope_smoke_sales'));
assert.ok(tables.every(t => t.tableName.startsWith('hankel_') && t.schemaName === 'public'));
assert.deepEqual(await api('GET', `${dsPath}/tables`), tables);
await api('GET', `${dsPath}/tables/hidden_scope_smoke_sales/columns?schemaName=public`, undefined, 403);
assert.equal((await api('GET', `${dsPath}/tables/hankel_scope_smoke_sales/columns?schemaName=public`)).length, 3);
const query = await api('POST', `${dsPath}/query`, {
  sql: 'SELECT region, amount FROM public.hankel_scope_smoke_sales WHERE amount >= :minimum ORDER BY amount DESC',
  params: { minimum: 50 }, maxRows: 2,
});
assert.equal(query.rowCount, 2);
assert.deepEqual(query.rows, [['East', 100], ['West', 75]]);
await api('POST', `${dsPath}/sql/probe`, { sql: 'SELECT COUNT(*) FROM public.hankel_scope_smoke_sales' });
for (const badSql of ['SELECT * FROM public.hidden_scope_smoke_sales',
  'SELECT * FROM scope_hidden.hankel_scope_smoke_sales', 'SELECT * FROM information_schema.tables']) {
  await api('POST', `${dsPath}/query`, { sql: badSql }, 403);
}
for (const badSql of ['DELETE FROM public.hankel_scope_smoke_sales', 'SELECT 1; SELECT 2',
  'SELECT * FROM another_database.public.hankel_scope_smoke_sales']) {
  await api('POST', `${dsPath}/query`, { sql: badSql }, 400);
}
const customSql = { datasourceId: ds.id, customSql: 'SELECT * FROM public.hankel_scope_smoke_sales' };
await api('POST', '/metrics/query', customSql, 403);

const tableMeta = { payload: { schemaName: 'public', tableName: 'hankel_scope_smoke_sales',
  docType: 'Scope Smoke Sales', columns: [{ name: 'region', type: 'varchar' },
    { name: 'amount', type: 'numeric' }, { name: 'quantity', type: 'integer' }] } };
const scopesBeforePublish = await api('GET', `${dsPath}/visible-scopes`);
await api('POST', `${dsPath}/meta/tables`, { payload: { schemaName: 'public', tableName: 'hidden_scope_smoke_sales' } }, 403);
assert.deepEqual(await api('GET', `${dsPath}/meta/tables/hidden_scope_smoke_sales`), []);
await api('POST', `${dsPath}/meta/tables`, tableMeta);
assert.deepEqual(await api('GET', `${dsPath}/visible-scopes`), scopesBeforePublish);
await api('POST', '/metrics/query', customSql);

const metricName = 'hankel_scope_smoke_total';
await api('POST', `${dsPath}/meta/metrics`, { objectType: 'metric_index', payload: {
  metric_name: metricName, display_name: 'Scope smoke total', source_type: 'cdp_postgres',
  source: { table_view: 'public.hankel_scope_smoke_sales' },
} });
await api('POST', `${dsPath}/meta/metrics`, { payload: {
  metric_name: metricName, display_name: 'Scope smoke total', source_type: 'cdp_postgres',
  source: { table_view: 'public.hankel_scope_smoke_sales' },
  calculation: { type: 'aggregate', aggregation: 'sum', measure: 'amount' },
  supported_dimensions: [{ dim_id: 'region', field_name: 'region' }],
} });
const semanticQuery = { datasourceId: ds.id, metrics: [metricName], groupBy: ['region'],
  orderBy: [{ field: 'region', direction: 'ASC' }] };
assert.deepEqual((await api('POST', '/metrics/query', semanticQuery)).rows, [['East', 150], ['West', 75]]);
const fullMeta = await api('GET', `${dsPath}/meta`);
assert.ok(fullMeta.index.metrics.some(m => m.metricName === metricName));
await api('PUT', `${dsPath}/visible-scopes/${scope.id}`, { ...rule, status: 0 });
const narrowedMeta = await api('GET', `${dsPath}/meta`);
assert.equal(narrowedMeta.index.metrics.length, 0);
assert.equal(narrowedMeta.index.tables.length, 0);
await api('POST', '/metrics/query', semanticQuery, 403);
await api('POST', '/metrics/query', customSql, 403);
await api('PUT', `${dsPath}/visible-scopes/${scope.id}`, rule);
assert.ok((await api('GET', `${dsPath}/meta`)).index.metrics.some(m => m.metricName === metricName));
await api('DELETE', `${dsPath}/meta/tables/hankel_scope_smoke_sales`);
assert.equal((await api('GET', `${dsPath}/visible-scopes`)).length, 1);
await api('POST', '/metrics/query', customSql, 403);
await api('POST', `${dsPath}/meta/tables`, tableMeta);
await api('DELETE', `${dsPath}/table-grants/${scope.id}`);
await api('POST', `${dsPath}/query`, { sql: 'SELECT 1' }, 403);
assert.equal((await api('GET', dsPath)).visibleScopeMode, 'RESTRICTED');
await api('PUT', dsPath, { ...config, visibleScopeMode: 'ALL' });
await api('POST', `${dsPath}/meta/tables`, { objectKey: 'scope_hidden.invalid_meta', payload: { schemaName: 'public' } }, 400);
assert.deepEqual(await api('GET', `${dsPath}/meta/tables/scope_hidden.invalid_meta`), []);
await api('POST', `${dsPath}/query`, { sql: 'SELECT * FROM public.hidden_scope_smoke_sales' });
await api('POST', `${dsPath}/query`, { sql: 'SELECT table_name FROM information_schema.tables', maxRows: 1 });
await api('POST', '/metrics/query', { datasourceId: ds.id, customSql: 'SELECT * FROM public.hidden_scope_smoke_sales' }, 403);
await api('POST', '/metrics/query', { datasourceId: ds.id, customSql: 'SELECT * FROM scope_hidden.hankel_scope_smoke_sales' }, 403);
assert.deepEqual((await api('POST', '/metrics/query', semanticQuery)).rows, [['East', 150], ['West', 75]]);
await api('PUT', dsPath, config);
assert.equal((await api('GET', dsPath)).visibleScopeMode, 'ALL');
await api('PUT', dsPath, { ...config, visibleScopeMode: 'invalid' }, 400);
await api('DELETE', dsPath);
console.log(JSON.stringify({ status: 'PASS', httpChecks: checks, datasourceId: ds.id,
  semanticRows: [['East', 150], ['West', 75]], database, server: base }, null, 2));
