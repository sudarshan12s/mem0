/// <reference types="jest" />

const mockLoadPeer = jest.fn();

jest.mock("../src/utils/load_peer", () => ({
  loadPeer: mockLoadPeer,
}));

const { OracleAIVectorSearch } = require("../src/vector_stores/oracledb");

const mockExecute = jest.fn();
const mockExecuteMany = jest.fn();
const mockCommit = jest.fn();
const mockRollback = jest.fn();
const mockClose = jest.fn();
const mockPoolGetConnection = jest.fn();
const mockPoolClose = jest.fn();
const mockCreatePool = jest.fn();
const mockGetConnection = jest.fn();
const mockConnection = {
  execute: mockExecute,
  executeMany: mockExecuteMany,
  commit: mockCommit,
  rollback: mockRollback,
  close: mockClose,
};
const mockPool = {
  getConnection: mockPoolGetConnection,
  close: mockPoolClose,
};

const mockDriver = {
  STRING: "STRING",
  DB_TYPE_VECTOR: "VECTOR",
  DB_TYPE_JSON: "JSON",
  OUT_FORMAT_ARRAY: "ARRAY",
  createPool: mockCreatePool,
  getConnection: mockGetConnection,
};

beforeEach(() => {
  jest.clearAllMocks();
  mockLoadPeer.mockResolvedValue(mockDriver);
  mockExecute.mockResolvedValue({ rows: [] });
  mockExecuteMany.mockResolvedValue({ rowsAffected: 1 });
  mockPoolGetConnection.mockResolvedValue(mockConnection);
  mockCreatePool.mockResolvedValue(mockPool);
  mockGetConnection.mockResolvedValue(mockConnection);
});

function createStore() {
  return new OracleAIVectorSearch({
    client: mockConnection,
    collectionName: "oracle_memories",
    embeddingModelDims: 3,
    doCreateIndex: false,
  });
}

describe("OracleAIVectorSearch", () => {
  it("creates a pool from connectionParams by default", async () => {
    const store = new OracleAIVectorSearch({
      connectionParams: {
        user: "oracle_user",
        password: "oracle_password",
        connectString: "localhost:1521/freepdb1",
        poolMin: 2,
        poolMax: 8,
      },
      collectionName: "pooled_memories",
      doCreateIndex: false,
    });

    await store.initialize();

    expect(mockCreatePool).toHaveBeenCalledWith(
      expect.objectContaining({
        user: "oracle_user",
        password: "oracle_password",
        connectString: "localhost:1521/freepdb1",
        poolMin: 2,
        poolMax: 8,
      }),
    );
    expect(mockPoolGetConnection).toHaveBeenCalled();
  });

  it("uses a direct connection when useConnectionPool is false", async () => {
    const store = new OracleAIVectorSearch({
      connectionParams: { user: "oracle_user", connectString: "db" },
      collectionName: "direct_memories",
      useConnectionPool: false,
      doCreateIndex: false,
    });

    await store.initialize();

    expect(mockGetConnection).toHaveBeenCalledWith({
      user: "oracle_user",
      connectString: "db",
    });
    expect(mockCreatePool).not.toHaveBeenCalled();
  });

  it("applies index and distance configuration to generated SQL", async () => {
    const store = new OracleAIVectorSearch({
      client: mockConnection,
      collectionName: "configured_memories",
      embeddingModelDims: 8,
      distanceMetric: "DOT",
      doCreateIndex: true,
      indexType: "IVF",
      indexName: "configured_idx",
      indexAccuracy: 90,
      indexParameters: {
        neighbor_partitions: 8,
        samples_per_partition: 16,
        min_vectors_per_partition: 2,
      },
    });

    await store.initialize();
    await store.search(new Array(8).fill(0.1));

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining(
        'CREATE VECTOR INDEX IF NOT EXISTS "configured_idx"',
      ),
    );
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining(
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE DOT WITH TARGET ACCURACY 90",
      ),
    );
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining(
        "PARAMETERS (type IVF, neighbor partitions 8, samples_per_partition 16, min_vectors_per_partition 2)",
      ),
    );
    expect(mockExecute).toHaveBeenLastCalledWith(
      expect.stringContaining("VECTOR_DISTANCE(vector, :query_vector, DOT)"),
      expect.any(Object),
      expect.any(Object),
    );
  });

  it("validates all constrained configuration values", () => {
    const baseConfig = {
      client: mockConnection,
      collectionName: "valid_memories",
    };

    expect(
      () => new OracleAIVectorSearch({ ...baseConfig, embeddingModelDims: 0 }),
    ).toThrow("embeddingModelDims");
    expect(
      () =>
        new OracleAIVectorSearch({
          ...baseConfig,
          distanceMetric: "INVALID" as "COSINE",
        }),
    ).toThrow("distance metric");
    expect(
      () =>
        new OracleAIVectorSearch({
          ...baseConfig,
          indexType: "INVALID" as "HNSW",
        }),
    ).toThrow("index type");
    expect(
      () =>
        new OracleAIVectorSearch({
          ...baseConfig,
          indexParameters: { unsupported: 1 },
        }),
    ).toThrow("Unsupported HNSW index parameter");
    expect(
      () => new OracleAIVectorSearch({ ...baseConfig, indexAccuracy: 101 }),
    ).toThrow("indexAccuracy");
  });

  it("renders validated HNSW index parameters", async () => {
    const store = new OracleAIVectorSearch({
      client: mockConnection,
      collectionName: "hnsw_memories",
      doCreateIndex: true,
      indexType: "HNSW",
      indexParameters: { neighbors: 32, efconstruction: 200 },
    });

    await store.initialize();

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining(
        "PARAMETERS (type HNSW, neighbors 32, efconstruction 200)",
      ),
    );
  });

  it("creates the vector and migration tables during initialization", async () => {
    const store = createStore();

    await store.initialize();

    expect(mockLoadPeer).toHaveBeenCalledWith(
      "oracledb",
      "Oracle AI Vector Search",
      expect.any(Function),
    );
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('CREATE TABLE IF NOT EXISTS "oracle_memories"'),
    );
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining("CREATE TABLE IF NOT EXISTS mem0_migrations"),
    );
    expect(mockCommit).toHaveBeenCalledTimes(2);
  });

  it("binds Float32 vectors and payloads using Oracle vector and JSON types", async () => {
    const store = createStore();
    await store.initialize();
    mockCommit.mockClear();

    await store.insert([[0.1, 0.2, 0.3]], ["memory-1"], [{ topic: "oracle" }]);

    expect(mockExecuteMany).toHaveBeenCalledWith(
      expect.stringContaining('MERGE INTO "oracle_memories"'),
      [["memory-1", expect.any(Float32Array), { topic: "oracle" }]],
      expect.objectContaining({
        bindDefs: [
          { type: "STRING", maxSize: 36 },
          { type: "VECTOR" },
          { type: "JSON" },
        ],
      }),
    );
    expect(mockCommit).toHaveBeenCalledTimes(1);
  });

  it("uses JSON_EXISTS filters and converts cosine distance to similarity", async () => {
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({
      rows: [["memory-1", { topic: "oracle" }, 0.125]],
    });
    const store = createStore();

    const results = await store.search([0.1, 0.2, 0.3], 1, {
      topic: "oracle",
    });

    expect(results).toEqual([
      { id: "memory-1", payload: { topic: "oracle" }, score: 0.875 },
    ]);
    expect(mockExecute).toHaveBeenLastCalledWith(
      expect.stringContaining("JSON_EXISTS(payload"),
      expect.objectContaining({ filter_0: "oracle", limit: 1 }),
      { outFormat: "ARRAY" },
    );
    expect(mockExecute.mock.calls.at(-1)?.[0]).not.toContain(
      "VECTOR_INDEX_TRANSFORM",
    );
  });

  it("keeps non-cosine Oracle metrics as distances", async () => {
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({
      rows: [["memory-1", { topic: "oracle" }, 0.125]],
    });
    const store = new OracleAIVectorSearch({
      client: mockConnection,
      collectionName: "oracle_memories",
      embeddingModelDims: 3,
      distanceMetric: "EUCLIDEAN",
      doCreateIndex: false,
    });

    const results = await store.search([0.1, 0.2, 0.3], 1);

    expect(results[0]?.score).toBe(0.125);
  });

  it("uses VECTOR_INDEX_TRANSFORM for unfiltered searches", async () => {
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({ rows: [] });
    mockExecute.mockResolvedValueOnce({ rows: [] });
    const store = createStore();

    await store.search([0.1, 0.2, 0.3], 1);

    expect(mockExecute.mock.calls.at(-1)?.[0]).toContain(
      'SELECT /*+ VECTOR_INDEX_TRANSFORM("oracle_memories") */',
    );
  });

  it("rejects unsafe collection and metadata filter keys", async () => {
    expect(
      () =>
        new OracleAIVectorSearch({
          client: mockConnection,
          collectionName: "memories; DROP TABLE users",
        }),
    ).toThrow("Invalid Oracle identifier");

    const store = createStore();
    await expect(
      store.search([0.1, 0.2, 0.3], 1, { "bad-key": "x" }),
    ).rejects.toThrow("Invalid Oracle metadata filter key");
  });

  it("is available through the vector store factory", () => {
    const { VectorStoreFactory } = require("../src/utils/factory");
    const store = VectorStoreFactory.create("oracle", {
      client: mockConnection,
      collectionName: "factory_memories",
    });

    expect(store).toBeInstanceOf(OracleAIVectorSearch);
  });

  it("relies on node-oracledb automatic rollback if creation fails", async () => {
    mockExecute.mockRejectedValueOnce(new Error("DDL Error"));
    const store = createStore();

    await expect(store.initialize()).rejects.toThrow("DDL Error");
    expect(mockRollback).not.toHaveBeenCalled();
  });

  it("does not close caller-provided connections", async () => {
    const store = createStore();
    await store.close();

    // A caller-provided connection remains owned by the caller. Connections
    // acquired from pools are released by withConnection(), and self-created
    // clients are closed by close().
    expect(mockClose).not.toHaveBeenCalled();
  });

  it("handles nested metadata filter keys", async () => {
    const store = createStore();
    await store.search([0.1, 0.2, 0.3], 1, { "user.id": "123" });

    expect(mockExecute).toHaveBeenLastCalledWith(
      expect.stringContaining(
        'JSON_EXISTS(payload, \'$."user"."id"?(@ == $filter_0)\'',
      ),
      expect.objectContaining({ filter_0: "123" }),
      expect.any(Object),
    );
  });

  it("supports pgvector-compatible comparison, membership, and boolean filters", async () => {
    const store = createStore();
    await store.search([0.1, 0.2, 0.3], 5, {
      $or: [
        { category: { in: ["books", "games"] } },
        { price: { gte: 10, lte: 20 } },
      ],
      status: { nin: ["archived", "deleted"] },
      title: { icontains: "Oracle" },
      author: ["Alice", "Bob"],
      hidden: { exists: false },
    });

    const [sql, binds] = mockExecute.mock.calls.at(-1);
    expect(sql).toContain(" OR ");
    expect(sql).toContain(" AND ");
    expect(sql).toContain("@ >= $filter_2");
    expect(sql).toContain("@ <= $filter_3");
    expect(sql).toContain("LOWER(JSON_VALUE(payload");
    expect(sql).toContain("NOT JSON_EXISTS(payload, '$.\"hidden\"')");
    expect(binds).toMatchObject({
      filter_0: "books",
      filter_1: "games",
      filter_2: 10,
      filter_3: 20,
      filter_4: "archived",
      filter_5: "deleted",
      filter_6: "%Oracle%",
      filter_7: "Alice",
      filter_8: "Bob",
      limit: 5,
    });
  });

  it("supports negated groups, between, and rejects invalid operators", async () => {
    const store = createStore();
    await store.search([0.1, 0.2, 0.3], 1, {
      $not: [{ rating: { between: [3, 5] } }, { category: "restricted" }],
    });

    const [sql, binds] = mockExecute.mock.calls.at(-1);
    expect(sql).toContain("NOT ((JSON_EXISTS(payload");
    expect(binds).toMatchObject({
      filter_0: 3,
      filter_1: 5,
      filter_2: "restricted",
    });

    await expect(
      store.search([0.1, 0.2, 0.3], 1, { rating: { unsupported: 1 } }),
    ).rejects.toThrow("Unsupported filter operator: unsupported");
  });

  it("supports every remaining pgvector filter operator and shorthand", async () => {
    const store = createStore();
    await store.search([0.1, 0.2, 0.3], 1, {
      $and: [{ visible: "*" }, { score: { $eq: 7, ne: 0, gt: 1, lt: 10 } }],
      title: { contains: "100%_coverage\\check" },
      published: { exists: true },
      emptyIn: { in: [] },
      emptyNin: { nin: [] },
    });

    const [sql, binds] = mockExecute.mock.calls.at(-1);
    expect(sql).toContain("JSON_EXISTS(payload, '$.\"visible\"')");
    expect(sql).toContain("@ == $filter_0");
    expect(sql).toContain("@ != $filter_1");
    expect(sql).toContain("@ > $filter_2");
    expect(sql).toContain("@ < $filter_3");
    expect(sql).toContain(
      "JSON_VALUE(payload, '$.\"title\"' RETURNING VARCHAR2(4000)) LIKE",
    );
    expect(sql).toContain("JSON_EXISTS(payload, '$.\"published\"')");
    expect(sql).toContain("1 = 0");
    expect(sql).toContain("1 = 1");
    expect(binds).toMatchObject({
      filter_0: 7,
      filter_1: 0,
      filter_2: 1,
      filter_3: 10,
      filter_4: "%100\\%\\_coverage\\\\check%",
    });
  });

  it("validates logical and operator filter values", async () => {
    const store = createStore();
    const invalidFilters = [
      { $or: { category: "books" } },
      { category: { in: "books" } },
      { category: { nin: "books" } },
      { price: { between: [1] } },
      { published: { exists: "yes" } },
      { $and: [] },
    ];

    for (const filters of invalidFilters) {
      await expect(store.search([0.1, 0.2, 0.3], 1, filters)).rejects.toThrow();
    }
  });

  it("applies translated filters to both list and count queries", async () => {
    const store = createStore();
    await store.list({ category: { eq: "books" } }, 10);

    const calls = mockExecute.mock.calls.slice(-2);
    expect(calls).toHaveLength(2);
    for (const [sql, binds] of calls) {
      expect(sql).toContain("JSON_EXISTS(payload");
      expect(binds).toMatchObject({ filter_0: "books" });
    }
  });

  it("supports array-wildcard paths without interpolating filter values", async () => {
    const store = createStore();
    await store.search([0.1, 0.2, 0.3], 1, {
      "users[*].role": "admin'); DROP TABLE mem0; --",
    });

    const [sql, binds] = mockExecute.mock.calls.at(-1);
    expect(sql).toContain(
      'JSON_EXISTS(payload, \'$."users"[*]."role"?(@ == $filter_0)\'',
    );
    expect(sql).not.toContain("DROP TABLE");
    expect(binds).toMatchObject({
      filter_0: "admin'); DROP TABLE mem0; --",
    });
  });
});
