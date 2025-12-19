-- System configuration table
CREATE TABLE IF NOT EXISTS system(
  team TEXT PRIMARY KEY,
  line_down_cost DOUBLE,
  plan_budget DOUBLE,
  space_factor DOUBLE,
  reserve_cost DOUBLE
);

-- Technology/user types
CREATE TABLE IF NOT EXISTS tech(
  tech_id INTEGER PRIMARY KEY,
  tech_name TEXT UNIQUE
);

-- Location priority settings
CREATE TABLE IF NOT EXISTS location_priority(
  location_priority INTEGER PRIMARY KEY,
  travel_time DOUBLE,
  reserve_allowed BOOLEAN
);

-- Storage locations
CREATE TABLE IF NOT EXISTS storage_location(
  location_id INTEGER PRIMARY KEY,
  location TEXT UNIQUE,
  floor_space DOUBLE,
  current_storage_floor_space DOUBLE,
  location_priority INTEGER,
  travel_time DOUBLE,
  reserve_allowed BOOLEAN,
  FOREIGN KEY (location_priority) REFERENCES location_priority(location_priority)
);

-- Tech usage of storage locations (many-to-many relationship), for UserLocationCompat[iloc, iuser]
-- a row exists if and only if UserLocationCompat[iloc, iuser] is true
CREATE TABLE IF NOT EXISTS tech_location_usage(
  tech_id INTEGER,
  location_id INTEGER,
  PRIMARY KEY (tech_id, location_id),
  FOREIGN KEY (tech_id) REFERENCES tech(tech_id),
  FOREIGN KEY (location_id) REFERENCES storage_location(location_id)
);

-- Storage types (bins, shelves, etc.)
CREATE TABLE IF NOT EXISTS storage_type(
  type_id INTEGER PRIMARY KEY,
  type TEXT UNIQUE,
  sqft_req DOUBLE,
  buy_cost DOUBLE,
  buy_invest DOUBLE,
  reloc_cost DOUBLE,
  reloc_invest DOUBLE,
  cubic_capacity_per_unit DOUBLE
);

-- Storage type availability in locations (many-to-many relationship)
-- a row exists if and only if a location has/uses a particular storage type
CREATE TABLE IF NOT EXISTS location_storage_type(
  location_id INTEGER,
  type_id INTEGER,
  code TEXT,
  type_current_units INTEGER,
  PRIMARY KEY (location_id, type_id),
  FOREIGN KEY (location_id) REFERENCES storage_location(location_id),
  FOREIGN KEY (type_id) REFERENCES storage_type(type_id)
);

CREATE TABLE IF NOT EXISTS size(
  size_id INTEGER,
  size TEXT,
  min_vol DOUBLE,
  total_cuft DOUBLE,
  PRIMARY KEY (size_id)
);


CREATE TABLE IF NOT EXISTS type_size_compat(
  type_id INTEGER,
  size_id INTEGER,
  max_sku_per_unit DOUBLE,
  PRIMARY KEY (type_id, size_id),
  FOREIGN KEY (type_id) REFERENCES storage_type(type_id),
  FOREIGN KEY (size_id) REFERENCES size(size_id)
);

CREATE TABLE IF NOT EXISTS parts(
  material TEXT PRIMARY KEY,
  priority TEXT,
  movement TEXT,
  size TEXT,
  orders DOUBLE,
  line_down_orders DOUBLE,
  total_stock DOUBLE,
  num_users DOUBLE,
  ASSEMBLY BOOLEAN,
  BODY BOOLEAN,
  PAINT BOOLEAN,
  UNK BOOLEAN,
  size_id INTEGER,
  updated_at TIMESTAMP DEFAULT now(),
  storage_type TEXT,
  FOREIGN KEY (size_id) REFERENCES size(size_id)
);

-- Cost centers with their associated users
CREATE TABLE IF NOT EXISTS cost_center(
  cost_center INTEGER,
  ASSEMBLY BOOLEAN,
  BODY BOOLEAN,
  PAINT BOOLEAN,
  UNK BOOLEAN,
  updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tech_map(
  value TEXT PRIMARY KEY,
  tech_name TEXT
);

CREATE TABLE IF NOT EXISTS i_sku(
  category_id INTEGER PRIMARY KEY,
  category TEXT,
  priority TEXT,
  movement TEXT,
  size TEXT,
  numSKU DOUBLE,
  total_stock DOUBLE,
  total_orders DOUBLE,
  line_down_orders DOUBLE,
  size_id INTEGER,
  min_vol DOUBLE,
  SKU_cubic_ft DOUBLE,
  FOREIGN KEY (size_id) REFERENCES size(size_id)
);

CREATE TABLE IF NOT EXISTS i_sku_user(
  category_id INTEGER,
  category TEXT,
  tech_id INTEGER,
  tech_name TEXT,
  numSKU DOUBLE,
  ld_orders_per_user DOUBLE,
  orders_per_user DOUBLE,
  PRIMARY KEY (category_id, tech_id),
  FOREIGN KEY (category_id) REFERENCES i_sku(category_id),
  FOREIGN KEY (tech_id) REFERENCES tech(tech_id)
);

CREATE TABLE IF NOT EXISTS i_sku_type(
  category_id INTEGER,
  category TEXT,
  type_id INTEGER,
  type TEXT,
  size_id INTEGER,
  cubic_capacity_per_unit DOUBLE,
  max_sku_per_unit DOUBLE,
  compatible BOOLEAN,
  penalty BOOLEAN,
  PRIMARY KEY (category_id, type_id, size_id),
  FOREIGN KEY (category_id) REFERENCES i_sku(category_id),
  FOREIGN KEY (type_id) REFERENCES storage_type(type_id),
  FOREIGN KEY (size_id) REFERENCES size(size_id)
);
