import { v4 as uuidv4 } from "uuid";
import type oracledb from "oracledb";

import type {
  OracleAIVectorSearchConfig,
  SearchFilters,
  VectorStoreResult,
} from "../types";
import { loadPeer } from "../utils/load_peer";
import type { VectorStore } from "./base";

const METADATA_KEY_PATTERN = /^[a-zA-Z0-9_.\[\]*]+$/;
const DISTANCE_METRICS = new Set([
  "EUCLIDEAN",
  "EUCLIDEAN_SQUARED",
  "COSINE",
  "DOT",
  "HAMMING",
  "MANHATTAN",
]);

type OracleDriver = typeof import("oracledb");
type OracleModule = OracleDriver & { default?: OracleDriver };
type OracleConnection = oracledb.Connection;
type OraclePool = oracledb.Pool;

/** Oracle Database AI Vector Search implementation for the OSS SDK. */
export class OracleAIVectorSearch implements VectorStore {
  private readonly config: OracleAIVectorSearchConfig;
  private readonly collectionName: string;
  private readonly indexName: string;
  private readonly dimension: number;
  private readonly distanceMetric: string;
  private readonly indexType: "HNSW" | "IVF";
  private driver?: OracleDriver;
  private pool?: OraclePool;
  private connection?: OracleConnection;
  private ownsClient = false;
  private initPromise?: Promise<void>;

  constructor(config: OracleAIVectorSearchConfig) {
    this.config = config;
    this.collectionName = quoteIdentifier(config.collectionName || "mem0");
    this.indexName = quoteIdentifier(
      config.indexName || `${config.collectionName || "mem0"}_VEC_IDX`,
    );
    this.dimension = config.dimension ?? config.embeddingModelDims ?? 1536;
    this.distanceMetric = (config.distanceMetric || "COSINE").toUpperCase();
    this.indexType = (config.indexType || "HNSW").toUpperCase() as
      | "HNSW"
      | "IVF";

    if (!DISTANCE_METRICS.has(this.distanceMetric)) {
      throw new Error(
        `Unsupported Oracle distance metric: ${this.distanceMetric}`,
      );
    }
    if (this.indexType !== "HNSW" && this.indexType !== "IVF") {
      throw new Error(`Unsupported Oracle index type: ${this.indexType}`);
    }
    validateIndexParameters(this.indexType, config.indexParameters);
    if (
      config.indexAccuracy !== undefined &&
      (!Number.isInteger(config.indexAccuracy) ||
        config.indexAccuracy < 1 ||
        config.indexAccuracy > 100)
    ) {
      throw new Error("indexAccuracy must be an integer between 1 and 100");
    }
    if (!Number.isInteger(this.dimension) || this.dimension <= 0) {
      throw new Error("embeddingModelDims must be a positive integer");
    }
  }

  async initialize(): Promise<void> {
    if (!this.initPromise) this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    const driver = await loadOracleDriver();
    this.driver = driver;
    if (this.config.client) {
      if (typeof this.config.client.getConnection === "function") {
        this.pool = this.config.client;
      } else {
        this.connection = this.config.client;
      }
    } else if (this.config.useConnectionPool !== false) {
      this.pool = await driver.createPool({
        poolMin: 1,
        poolMax: 4,
        ...this.config.connectionParams,
      } as oracledb.PoolAttributes);
      this.ownsClient = true;
    } else {
      this.connection = await driver.getConnection(
        this.config.connectionParams as oracledb.ConnectionAttributes,
      );
      this.ownsClient = true;
    }
    await this.createCol();
    await this.createMigrationTable();
  }

  private async withConnection<T>(
    operation: (connection: OracleConnection) => Promise<T>,
    commit = false,
    skipInitialize = false,
  ): Promise<T> {
    if (!skipInitialize) await this.initialize();
    const connection = this.pool
      ? await this.pool.getConnection()
      : this.connection;
    if (!connection) throw new Error("Oracle connection is not initialized");
    try {
      const result = await operation(connection);
      if (commit) await connection.commit();
      return result;
    } finally {
      // the rollback happens automatically on error.
      if (this.pool) await connection.close();
    }
  }

  private async createCol(): Promise<void> {
    await this.withConnection(
      async (connection) => {
        await connection.execute(
          `CREATE TABLE IF NOT EXISTS ${this.collectionName} (
          id VARCHAR2(36) PRIMARY KEY,
          vector VECTOR(${this.dimension}),
          payload JSON
        )`,
        );
        if (this.config.doCreateIndex !== false) {
          await connection.execute(this.createIndexDdl());
        }
      },
      true,
      true,
    );
  }

  private async createMigrationTable(): Promise<void> {
    await this.withConnection(
      (connection) =>
        connection.execute(
          "CREATE TABLE IF NOT EXISTS mem0_migrations (id NUMBER DEFAULT 1 PRIMARY KEY CHECK (id = 1),user_id VARCHAR2(255) NOT NULL)",
        ),
      true,
      true,
    );
  }

  private createIndexDdl(): string {
    const organization =
      this.indexType === "HNSW"
        ? "INMEMORY NEIGHBOR GRAPH"
        : "NEIGHBOR PARTITIONS";
    const accuracy = this.config.indexAccuracy
      ? ` WITH TARGET ACCURACY ${this.config.indexAccuracy}`
      : "";
    const parameters = this.indexParameters();
    const parameterClause = parameters ? ` PARAMETERS (${parameters})` : "";
    return (
      `CREATE VECTOR INDEX IF NOT EXISTS ${this.indexName} ON ${this.collectionName} (vector) ` +
      `ORGANIZATION ${organization} DISTANCE ${this.distanceMetric}${accuracy}${parameterClause}`
    );
  }

  private indexParameters(): string {
    const parameters = this.config.indexParameters;
    if (!parameters || Object.keys(parameters).length === 0) return "";
    const allowed =
      this.indexType === "HNSW"
        ? ["neighbors", "efconstruction"]
        : [
            "neighbor_partitions",
            "samples_per_partition",
            "min_vectors_per_partition",
          ];
    const parts = [`type ${this.indexType}`];
    for (const key of allowed) {
      const value = parameters[key];
      if (value === undefined) continue;
      if (!Number.isInteger(value) || value < 0) {
        throw new Error(
          `indexParameters.${key} must be a non-negative integer`,
        );
      }
      const label = key === "neighbor_partitions" ? "neighbor partitions" : key;
      parts.push(`${label} ${value}`);
    }
    const extra = Object.keys(parameters).filter(
      (key) => !allowed.includes(key),
    );
    if (extra.length) {
      throw new Error(
        `Unsupported ${this.indexType} index parameters: ${extra.join(", ")}`,
      );
    }
    return parts.join(", ");
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, any>[] = [],
  ): Promise<void> {
    if (!vectors.length) return;

    await this.initialize();

    const sql = `
    MERGE INTO ${this.collectionName} target
    USING (
      SELECT 
        :1 AS id, 
        :2 AS vector, 
        :3 AS payload 
      FROM dual
    ) src
    ON (target.id = src.id)
    WHEN MATCHED THEN
      UPDATE SET 
        target.vector = src.vector, 
        target.payload = src.payload
    WHEN NOT MATCHED THEN
      INSERT (id, vector, payload)
      VALUES (src.id, src.vector, src.payload)
  `;

    // Explicit bindDefs prevent driver type scanning across batch elements
    const bindDefs = [
      { type: this.driver!.STRING, maxSize: 36 },
      { type: this.driver!.DB_TYPE_VECTOR },
      { type: this.driver!.DB_TYPE_JSON },
    ];

    // Map positional arguments into arrays matching :1, :2, :3
    const binds = vectors.map((vec, i) => [
      ids[i],
      Float32Array.from(vec),
      payloads[i] ?? {},
    ]);

    await this.withConnection(async (connection) => {
      const result = await connection.executeMany(sql, binds, {
        autoCommit: false,
        batchErrors: true,
        bindDefs,
      });

      // Handle partial row failures -> Strict All-or-None
      if (result.batchErrors && result.batchErrors.length > 0) {
        console.error(
          `[OracleAIVectorSearch] Batch insert failed with ${result.batchErrors.length} row error(s). Rolling back transaction.`,
        );

        for (const err of result.batchErrors) {
          const failedId =
            typeof err.offset === "number"
              ? (ids[err.offset] ?? "unknown")
              : "unknown";
          console.error(
            `  - Row index [${err.offset}] (ID: ${failedId}): ${err.message}`,
          );
        }

        // Throwing aborts the callback -> withConnection performs connection.rollback()
        throw new Error(
          `Batch insert failed on ${result.batchErrors.length} record(s). Transaction rolled back.`,
        );
      }

      await connection.commit();
    }, true);
  }

  async search(
    query: number[],
    topK = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const { clause, binds } = buildFilters(filters);
    // Oracle's vector index transform can be used only when there is no
    // metadata predicate. Applying it to a filtered query may select the
    // approximate top-k rows before the filter is evaluated.
    const selectClause = clause
      ? "SELECT"
      : `SELECT /*+ VECTOR_INDEX_TRANSFORM(${this.collectionName}) */`;
    const sql = `${selectClause} id, payload, VECTOR_DISTANCE(vector, :query_vector, ${this.distanceMetric}) distance
      FROM ${this.collectionName} ${clause}
      ORDER BY VECTOR_DISTANCE(vector, :query_vector, ${this.distanceMetric})
      FETCH FIRST :limit ROWS ONLY`;
    return this.withConnection(async (connection) => {
      const result = await connection.execute<[string, unknown, number]>(
        sql,
        oracleBindParameters({
          query_vector: Float32Array.from(query),
          limit: topK,
          ...binds,
        }),
        { outFormat: this.driver!.OUT_FORMAT_ARRAY },
      );
      return (result.rows || []).map((row) => {
        const distance = Number(row[2]);
        return {
          id: row[0],
          payload: parsePayload(row[1]),
          // Match other vector stores cosine-score convention: larger is more similar.
          // Other Oracle metrics remain distances because they have no common,
          // lossless conversion to a similarity score.
          score:
            this.distanceMetric === "COSINE"
              ? Math.max(0, Math.min(1, 1 - distance))
              : distance,
        };
      });
    });
  }

  async get(vectorId: string): Promise<VectorStoreResult | null> {
    return this.withConnection(async (connection) => {
      const result = await connection.execute<[string, unknown]>(
        `SELECT id, payload FROM ${this.collectionName} WHERE id = :id`,
        { id: vectorId },
        { outFormat: this.driver!.OUT_FORMAT_ARRAY },
      );
      const row = result.rows?.[0] as [string, unknown] | undefined;
      return row ? { id: row[0], payload: parsePayload(row[1]) } : null;
    });
  }

  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, any>,
  ): Promise<void> {
    await this.withConnection(async (connection) => {
      await connection.execute(
        `UPDATE ${this.collectionName} SET vector = :vector, payload = :payload WHERE id = :id`,
        oracleBindParameters({
          vector: {
            val: Float32Array.from(vector),
            type: this.driver!.DB_TYPE_VECTOR,
          },
          payload: { val: payload, type: this.driver!.DB_TYPE_JSON },
          id: vectorId,
        }),
      );
    }, true);
  }

  async delete(vectorId: string): Promise<void> {
    await this.withConnection(
      (connection) =>
        connection.execute(
          `DELETE FROM ${this.collectionName} WHERE id = :id`,
          {
            id: vectorId,
          },
        ),
      true,
    );
  }

  async deleteCol(): Promise<void> {
    await this.withConnection(
      (connection) =>
        connection.execute(`DROP TABLE ${this.collectionName} PURGE`),
      true,
    );
  }

  async list(
    filters?: SearchFilters,
    topK = 100,
  ): Promise<[VectorStoreResult[], number]> {
    const { clause, binds } = buildFilters(filters);
    return this.withConnection(async (connection) => {
      const [rowsResult, countResult] = await Promise.all([
        connection.execute<[string, unknown]>(
          `SELECT id, payload FROM ${this.collectionName} ${clause} FETCH FIRST :limit ROWS ONLY`,
          oracleBindParameters({ limit: topK, ...binds }),
          { outFormat: this.driver!.OUT_FORMAT_ARRAY },
        ),
        connection.execute<[number]>(
          `SELECT COUNT(*) FROM ${this.collectionName} ${clause}`,
          oracleBindParameters(binds),
          { outFormat: this.driver!.OUT_FORMAT_ARRAY },
        ),
      ]);
      const rows = (rowsResult.rows || []).map((row) => ({
        id: row[0],
        payload: parsePayload(row[1]),
      }));
      return [rows, Number(countResult.rows?.[0]?.[0] || 0)];
    });
  }

  async getUserId(): Promise<string> {
    const generated = uuidv4();
    return this.withConnection(async (connection) => {
      // Single atomic MERGE handles concurrent race conditions
      await connection.execute(
        `MERGE INTO mem0_migrations m
       USING (SELECT 1 AS id, :generated_id AS user_id FROM dual) src
       ON (m.id = src.id)
       WHEN NOT MATCHED THEN
         INSERT (id, user_id) VALUES (src.id, src.user_id)`,
        { generated_id: generated },
      );

      const result = await connection.execute<[string]>(
        "SELECT user_id FROM mem0_migrations WHERE id = 1",
        [],
        { outFormat: this.driver!.OUT_FORMAT_ARRAY },
      );

      return String(result.rows![0][0]);
    }, true);
  }

  async setUserId(userId: string): Promise<void> {
    await this.withConnection(async (connection) => {
      await connection.execute(
        `MERGE INTO mem0_migrations m
       USING (SELECT 1 AS id, :user_id AS user_id FROM dual) src
       ON (m.id = src.id)
       WHEN MATCHED THEN
         UPDATE SET m.user_id = src.user_id
       WHEN NOT MATCHED THEN
         INSERT (id, user_id) VALUES (src.id, src.user_id)`,
        { user_id: userId },
      );
    }, true);
  }

  async close(): Promise<void> {
    if (!this.ownsClient) return;
    if (this.pool) await this.pool.close();
    if (this.connection) await this.connection.close();
  }
}

function quoteIdentifier(identifier: string): string {
  const name = identifier.trim();

  const validateRegex =
    /^(?:"[^"]+"|[A-Za-z][A-Za-z0-9_$#]*)(?:\.(?:"[^"]+"|[A-Za-z][A-Za-z0-9_$#]*))*$/;
  if (!validateRegex.test(name)) {
    throw new Error(`Invalid Oracle identifier: ${identifier}`);
  }

  // extracts parts of the identifier with quoted and unquoted.
  const matchRegex = /"([^"]+)"|([A-Za-z][A-Za-z0-9_$#]*)/g;
  const groups = [];

  for (const match of name.matchAll(matchRegex)) {
    groups.push(match[1] || match[2]);
  }
  const quotedParts = groups.map((g) => `"${g}"`);
  return quotedParts.join(".");
}

type FilterState = {
  binds: Record<string, unknown>;
  nextParameter: number;
};

function buildFilters(filters?: SearchFilters): {
  clause: string;
  binds: Record<string, unknown>;
} {
  if (!filters || Object.keys(filters).length === 0) {
    return { clause: "", binds: {} };
  }

  const state: FilterState = { binds: {}, nextParameter: 0 };
  return {
    clause: `WHERE ${buildFilterConditions(filters, state)}`,
    binds: state.binds,
  };
}

function buildFilterConditions(
  filters: Record<string, any>,
  state: FilterState,
): string {
  const clauses: string[] = [];

  for (const [key, value] of Object.entries(filters)) {
    if (key === "$or" || key === "$and" || key === "$not") {
      if (!Array.isArray(value) || value.some((item) => !isPlainObject(item))) {
        throw new Error(`${key} filter must be an array of filter objects`);
      }
      const children = value.map((item) => buildFilterConditions(item, state));
      if (children.length === 0) continue;
      if (key === "$or") clauses.push(`(${children.join(" OR ")})`);
      else if (key === "$and") clauses.push(`(${children.join(" AND ")})`);
      else clauses.push(`NOT (${children.join(" OR ")})`);
      continue;
    }

    validateMetadataKey(key);
    if (value === "*") {
      clauses.push(`JSON_EXISTS(payload, '${jsonPath(key)}')`);
    } else if (Array.isArray(value)) {
      clauses.push(buildInCondition(key, value, false, state));
    } else if (isPlainObject(value)) {
      for (const [operator, operand] of Object.entries(value)) {
        clauses.push(buildOperatorCondition(key, operator, operand, state));
      }
    } else {
      clauses.push(buildComparison(key, "==", value, state));
    }
  }

  if (clauses.length === 0) throw new Error("Filter object must not be empty");
  return clauses.length === 1 ? clauses[0] : `(${clauses.join(" AND ")})`;
}

function buildOperatorCondition(
  key: string,
  operator: string,
  value: unknown,
  state: FilterState,
): string {
  const normalized = operator.startsWith("$") ? operator.slice(1) : operator;
  switch (normalized) {
    case "eq":
      return buildComparison(key, "==", value, state);
    case "ne":
      return buildComparison(key, "!=", value, state);
    case "gt":
    case "gte":
    case "lt":
    case "lte":
      return buildComparison(
        key,
        comparisonOperator(normalized as "gt" | "gte" | "lt" | "lte"),
        value,
        state,
      );
    case "in":
      if (!Array.isArray(value)) throw new Error("$in requires an array value");
      return buildInCondition(key, value, false, state);
    case "nin":
      if (!Array.isArray(value))
        throw new Error("$nin requires an array value");
      return buildInCondition(key, value, true, state);
    case "between":
      if (!Array.isArray(value) || value.length !== 2) {
        throw new Error("$between requires a two-element array");
      }
      return `(${buildComparison(key, ">=", value[0], state)} AND ${buildComparison(key, "<=", value[1], state)})`;
    case "exists":
      if (typeof value !== "boolean")
        throw new Error("$exists requires a boolean value");
      return `${value ? "" : "NOT "}JSON_EXISTS(payload, '${jsonPath(key)}')`;
    case "contains":
    case "icontains":
      return buildContainsCondition(
        key,
        value,
        normalized === "icontains",
        state,
      );
    default:
      throw new Error(`Unsupported filter operator: ${operator}`);
  }
}

function comparisonOperator(
  operator: "gt" | "gte" | "lt" | "lte",
): ">" | ">=" | "<" | "<=" {
  const operators = { gt: ">", gte: ">=", lt: "<", lte: "<=" } as const;
  return operators[operator];
}

function validateIndexParameters(
  indexType: "HNSW" | "IVF",
  parameters: Record<string, number> | undefined,
): void {
  if (!parameters) return;
  const ranges =
    indexType === "HNSW"
      ? { neighbors: [2, 2048], efconstruction: [1, 65535] }
      : {
          neighbor_partitions: [1, 10_000_000],
          samples_per_partition: [1, Infinity],
          min_vectors_per_partition: [0, Infinity],
        };
  for (const [key, value] of Object.entries(parameters)) {
    const range = ranges[key as keyof typeof ranges];
    if (!range) {
      throw new Error(`Unsupported ${indexType} index parameter: ${key}`);
    }
    if (!Number.isInteger(value) || value < range[0] || value > range[1]) {
      throw new Error(
        `indexParameters.${key} must be an integer between ${range[0]} and ${range[1]}`,
      );
    }
  }
}

function buildComparison(
  key: string,
  operator: "==" | "!=" | ">" | ">=" | "<" | "<=",
  value: unknown,
  state: FilterState,
): string {
  const parameter = addBind(value, state);
  return `JSON_EXISTS(payload, '${jsonPath(key)}?(@ ${operator} $${parameter})' PASSING :${parameter} AS "${parameter}")`;
}

function buildInCondition(
  key: string,
  values: unknown[],
  negate: boolean,
  state: FilterState,
): string {
  if (values.length === 0) return negate ? "1 = 1" : "1 = 0";
  const comparisons = values.map((value) =>
    buildComparison(key, "==", value, state),
  );
  return `(${comparisons.map((condition) => (negate ? `NOT (${condition})` : condition)).join(negate ? " AND " : " OR ")})`;
}

function buildContainsCondition(
  key: string,
  value: unknown,
  insensitive: boolean,
  state: FilterState,
): string {
  const escaped = String(value)
    .replace(/\\/g, "\\\\")
    .replace(/%/g, "\\%")
    .replace(/_/g, "\\_");
  const parameter = addBind(`%${escaped}%`, state);
  const expression = `JSON_VALUE(payload, '${jsonPath(key)}' RETURNING VARCHAR2(4000))`;
  return insensitive
    ? `LOWER(${expression}) LIKE LOWER(:${parameter}) ESCAPE '\\'`
    : `${expression} LIKE :${parameter} ESCAPE '\\'`;
}

function addBind(value: unknown, state: FilterState): string {
  const parameter = `filter_${state.nextParameter++}`;
  state.binds[parameter] = value;
  return parameter;
}

function validateMetadataKey(key: string): void {
  if (!METADATA_KEY_PATTERN.test(key)) {
    throw new Error(`Invalid Oracle metadata filter key: ${key}`);
  }
}

function jsonPath(key: string): string {
  return (
    "$" +
    key
      .split(".")
      .map((part) =>
        part.endsWith("[*]") ? `."${part.slice(0, -3)}"[*]` : `."${part}"`,
      )
      .join("")
  );
}

function isPlainObject(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parsePayload(payload: unknown): Record<string, any> {
  if (payload === null || payload === undefined) return {};
  if (typeof payload === "object" && !Buffer.isBuffer(payload)) {
    return payload as Record<string, any>;
  }
  return JSON.parse(
    Buffer.isBuffer(payload) ? payload.toString("utf8") : String(payload),
  );
}

function oracleBindParameters(
  values: Record<string, unknown>,
): oracledb.BindParameters {
  // @types/oracledb does not yet include Float32Array as a VECTOR bind value.
  return values as unknown as oracledb.BindParameters;
}

function vectorBindParameters(
  id: string,
  vector: number[],
  payload: Record<string, any>,
): oracledb.BindParameters {
  return oracleBindParameters({
    id,
    vector: Float32Array.from(vector),
    payload,
  });
}

async function loadOracleDriver(): Promise<OracleDriver> {
  // Keep the optional peer lazy-loaded until the Oracle store is initialized.
  const module = (await loadPeer(
    "oracledb",
    "Oracle AI Vector Search",
    () => import("oracledb"),
  )) as OracleModule;
  return module.default || module;
}
