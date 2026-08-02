"""In-memory Firestore fakes shared by the test suite (no live database)."""

from google.cloud.firestore_v1.transforms import Increment


class FakeSnapshot:
    def __init__(self, doc_id, data, reference=None):
        self.id = doc_id
        self._data = data or {}
        self.exists = data is not None
        self.reference = reference

    def get(self, field):
        return self._data.get(field)

    def to_dict(self):
        return dict(self._data)


class FakeDocRef:
    def __init__(self, db, coll_path, doc_id):
        self._db = db
        self._coll_path = coll_path
        self.id = doc_id

    @property
    def _store(self):
        return self._db.store_for(self._coll_path)

    def get(self):
        data = self._store.get(self.id)
        return FakeSnapshot(self.id, data, self)

    def set(self, data, merge=False):
        _apply_write(self._store, self.id, data, merge)

    def collection(self, name):
        return FakeCollection(self._db, f"{self._coll_path}/{self.id}/{name}")


class FakeQuery:
    def __init__(self, db, coll_path, filters=None, order=None, limit_n=None):
        self._db = db
        self._coll_path = coll_path
        self._filters = filters or []
        self._order = order
        self._limit_n = limit_n

    def where(self, field, op, value):
        return FakeQuery(
            self._db, self._coll_path, self._filters + [(field, op, value)],
            self._order, self._limit_n,
        )

    def select(self, fields):
        return self

    def order_by(self, field, direction=None):
        return FakeQuery(self._db, self._coll_path, self._filters, (field, direction), self._limit_n)

    def limit(self, n):
        return FakeQuery(self._db, self._coll_path, self._filters, self._order, n)

    def _matches(self, data):
        for field, op, value in self._filters:
            actual = data.get(field)
            if op == "==":
                if actual != value:
                    return False
            elif op == ">=":
                if actual is None or actual < value:
                    return False
            else:
                raise NotImplementedError(f"FakeQuery op {op}")
        return True

    def stream(self):
        store = self._db.store_for(self._coll_path)
        results = [
            FakeSnapshot(doc_id, data, FakeDocRef(self._db, self._coll_path, doc_id))
            for doc_id, data in list(store.items())
            if self._matches(data)
        ]
        if self._order:
            field, direction = self._order
            reverse = direction == "DESCENDING"
            results.sort(key=lambda snap: snap.get(field), reverse=reverse)
        if self._limit_n is not None:
            results = results[: self._limit_n]
        return results


class FakeCollection(FakeQuery):
    def __init__(self, db, coll_path):
        super().__init__(db, coll_path)
        self.added = []

    def document(self, doc_id):
        return FakeDocRef(self._db, self._coll_path, doc_id)

    def add(self, data):
        store = self._db.store_for(self._coll_path)
        doc_id = f"auto{len(store)}"
        store[doc_id] = dict(data)
        self.added.append(data)


class FakeBatch:
    def __init__(self, db):
        self._db = db
        self._ops = []

    def set(self, doc_ref, data, merge=False):
        self._ops.append(("set", doc_ref, data, merge))

    def update(self, doc_ref, data):
        self._ops.append(("update", doc_ref, data, True))

    def commit(self):
        assert len(self._ops) <= 500, "Firestore batch exceeded 500 writes"
        for op, doc_ref, data, merge in self._ops:
            if op == "update" and doc_ref.id not in doc_ref._store:
                raise KeyError(f"update on missing document {doc_ref.id}")
            _apply_write(doc_ref._store, doc_ref.id, data, merge)
        self._db.commit_sizes.append(len(self._ops))
        self._ops = []


class FakeDB:
    def __init__(self, existing_contacts=None):
        self.stores = {"contacts": dict(existing_contacts or {})}
        self.collections = {}
        self.commit_sizes = []

    def store_for(self, coll_path):
        return self.stores.setdefault(coll_path, {})

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = FakeCollection(self, name)
        return self.collections[name]

    def batch(self):
        return FakeBatch(self)


def _apply_write(store, doc_id, data, merge):
    resolved = {}
    existing = store.get(doc_id, {})
    for key, value in data.items():
        if isinstance(value, Increment):
            base = existing.get(key)
            resolved[key] = (base if isinstance(base, (int, float)) else 0) + value.value
        else:
            resolved[key] = value

    if merge and doc_id in store:
        store[doc_id].update(resolved)
    else:
        store[doc_id] = resolved


def make_phone(i):
    """Valid, strict-passing +2547XXXXXXXX numbers."""
    return f"+2547{10000000 + i:08d}"
