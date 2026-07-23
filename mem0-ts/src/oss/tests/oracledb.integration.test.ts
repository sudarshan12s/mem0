/// <reference types="jest" />

import { OracleAIVectorSearch } from "../src/vector_stores/oracledb";

const oracleConfig = {
  user: process.env.ORACLE_USERNAME,
  password: process.env.ORACLE_PASSWORD,
  connectString: process.env.ORACLE_DSN,
};
const hasOracleCredentials = Object.values(oracleConfig).every(Boolean);
const describeOracle = hasOracleCredentials ? describe : describe.skip;

describeOracle("OracleAIVectorSearch integration", () => {
  let store: OracleAIVectorSearch;
  const collectionName = `MEM0_TS_ORACLE_${Date.now()}_${Math.floor(Math.random() * 100000)}`;

  beforeAll(async () => {
    store = new OracleAIVectorSearch({
      connectionParams: oracleConfig,
      collectionName,
      embeddingModelDims: 3,
      distanceMetric: "COSINE",
      useConnectionPool: true,
      // Index creation is covered by unit tests. Keeping it off makes the
      // CRUD/filter integration test independent of index-training state.
      doCreateIndex: false,
    });
    await store.initialize();
  });

  afterAll(async () => {
    if (!store) return;
    try {
      await store.deleteCol();
    } finally {
      await store.close();
    }
  });

  it("inserts, searches, filters, updates, lists, and deletes vectors", async () => {
    await store.insert(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      ["oracle-1", "oracle-2", "oracle-3"],
      [
        { category: "books", rating: 5, tags: ["ai"] },
        { category: "books", rating: 3, tags: ["db"] },
        { category: "music", rating: 4, tags: ["ai"] },
      ],
    );

    const unfilteredResults = await store.search([1, 0, 0], 2);
    expect(unfilteredResults.map((result) => result.id)).toContain("oracle-1");
    expect(unfilteredResults[0]?.score).toBeCloseTo(1);

    const searchResults = await store.search([1, 0, 0], 5, {
      category: "books",
      rating: { gte: 4 },
    });
    expect(searchResults).toHaveLength(1);
    expect(searchResults[0]).toMatchObject({
      id: "oracle-1",
      payload: { category: "books", rating: 5 },
    });
    expect(searchResults[0]?.score).toBeCloseTo(1);

    // MERGE should update an existing vector rather than add a duplicate row.
    await store.insert(
      [[0, 0, 0.95]],
      ["oracle-3"],
      [{ category: "music", rating: 5, tags: ["ai", "upserted"] }],
    );
    expect(await store.get("oracle-3")).toMatchObject({
      payload: { category: "music", rating: 5, tags: ["ai", "upserted"] },
    });

    await store.update("oracle-1", [0.9, 0.1, 0], {
      category: "books",
      rating: 6,
      tags: ["ai", "updated"],
    });
    expect(await store.get("oracle-1")).toMatchObject({
      payload: { category: "books", rating: 6 },
    });
    expect(await store.get("missing-vector")).toBeNull();

    const [listed, count] = await store.list({ category: { in: ["books"] } });
    expect(count).toBe(2);
    expect(listed.map((result) => result.id).sort()).toEqual([
      "oracle-1",
      "oracle-2",
    ]);

    const [limited, total] = await store.list(undefined, 1);
    expect(limited).toHaveLength(1);
    expect(total).toBe(3);

    await store.delete("oracle-2");
    expect(await store.get("oracle-2")).toBeNull();
  });

  it("persists the configured user ID", async () => {
    const originalUserId = await store.getUserId();
    const configuredUserId = `integration-user-${Date.now()}`;

    try {
      await store.setUserId(configuredUserId);
      await expect(store.getUserId()).resolves.toBe(configuredUserId);
    } finally {
      await store.setUserId(originalUserId);
    }
  });

  it("supports direct connections and preserves non-cosine distance scores", async () => {
    const directCollectionName = `${collectionName}_EUCLIDEAN`;
    const directStore = new OracleAIVectorSearch({
      connectionParams: oracleConfig,
      collectionName: directCollectionName,
      embeddingModelDims: 3,
      distanceMetric: "EUCLIDEAN",
      useConnectionPool: false,
      doCreateIndex: false,
    });

    try {
      await directStore.initialize();
      await directStore.insert([[1, 0, 0]], ["direct-1"], [{ kind: "direct" }]);

      const results = await directStore.search([0, 1, 0], 1);
      expect(results).toHaveLength(1);
      expect(results[0]).toMatchObject({
        id: "direct-1",
        payload: { kind: "direct" },
      });
      expect(results[0]?.score).toBeCloseTo(Math.sqrt(2));
    } finally {
      await directStore.deleteCol();
      await directStore.close();
    }
  });
});
