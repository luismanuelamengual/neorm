import { DataTable } from '../database/DataTable'
import { PaginationResult } from '../database/PaginationResult'
import { Field, SelectQuery } from '../database/query'
import { Condition, ConditionGroup, ExistsSubquery } from '../database/query/conditions'
import { JoinType } from '../database/query/features/HasJoins'
import { OrderByDirection } from '../database/query/features/HasOrderByFields'
import { EntityRepository, getGlobalScopes, Repository } from './EntityRepository'

/** A single eager-load entry: the dot-notation path and an optional constraint callback. */
export type WithEntry = {
  path: string
  callback?: (query: EntityQuery<any>) => void
}

/**
 * A whereHas subquery whose related entity's global scopes have not been
 * applied yet. `conditions` is the EXISTS' own condition group, already wired
 * into the outer statement; `constraints` is the live group the sub-query
 * writes its conditions into, attached to `conditions` only once it is known to
 * hold something (an empty group would compile to an empty `()`).
 */
type PendingRelationQuery = {
  query: EntityQuery<any>
  conditions: ConditionGroup
  constraints: ConditionGroup
}

/**
 * Chainable query builder for Entities. Wraps DataTable and hydrates rows
 * into typed entity instances when a terminal method is called.
 *
 * Constructed by EntityRepository.query() — do not instantiate directly.
 */
export class EntityQuery<T> {
  private _repository: EntityRepository<T>
  private _table: DataTable
  private _withs: WithEntry[] = []
  private _scopesApplied = false
  private _excludedScopes: Set<string> = new Set()
  private _skipAllScopes = false
  /**
   * Subqueries built by whereHas/orWhereHas, waiting for their own entity's
   * global scopes. They cannot be resolved when the EXISTS is built, because
   * whereHas is a synchronous chainable and a scope may be asynchronous — so
   * they are held here and resolved by _applyGlobalScopes, which already runs
   * (once, awaited) right before the statement is compiled. See _applyWhereHas.
   */
  private _pendingRelationQueries: PendingRelationQuery[] = []

  constructor(repository: EntityRepository<T>, table: DataTable) {
    this._repository = repository
    this._table = table
  }

  // ── Global scope control ─────────────────────────────────────────────────────

  /**
   * Excludes one or more named global scopes from being applied to this query.
   *
   *   Product.withoutGlobalScope('active').get()
   */
  public withoutGlobalScope(...names: string[]): this {
    for (const name of names) {
      this._excludedScopes.add(name)
    }

    return this
  }

  /**
   * Disables all global scopes for this query.
   *
   *   Product.withoutGlobalScopes().get()
   */
  public withoutGlobalScopes(): this {
    this._skipAllScopes = true

    return this
  }

  /** Applies all registered global scopes that have not been excluded. Called lazily before terminal methods. */
  private async _applyGlobalScopes(): Promise<void> {
    if (this._scopesApplied) {
      return
    }

    this._scopesApplied = true

    if (this._skipAllScopes) {
      return
    }

    const entityClass = (this._repository as any)._entityClass
    const scopes = getGlobalScopes(entityClass)

    for (const [name, scopeFn] of scopes) {
      if (!this._excludedScopes.has(name)) {
        await scopeFn(this)
      }
    }

    await this._resolvePendingRelationQueries()
  }

  /**
   * Applies the pending whereHas subqueries' own global scopes and folds their
   * conditions into the EXISTS that is already part of this statement.
   *
   * This is what makes `whereHas('posts')` mean "has posts that are visible to
   * you" rather than "has a row in the posts table": a global scope states an
   * invariant about the entity, so a relation constraint has to honour it the
   * same way a direct query does — and the same way `with()` already does when
   * it eager-loads that relation. A callback that wants the raw table opts out
   * from inside, with the sub-query's own `withoutGlobalScope(s)`.
   *
   * Resolution is recursive: applying a sub-query's scopes resolves ITS pending
   * subqueries in turn, so a nested whereHas is scoped at every level.
   */
  private async _resolvePendingRelationQueries(): Promise<void> {
    // Drained rather than iterated: a scope is free to add another whereHas,
    // and that one has to be resolved too.
    while (this._pendingRelationQueries.length > 0) {
      const pending = this._pendingRelationQueries.splice(0)

      for (const { query, conditions, constraints } of pending) {
        await query._applyGlobalScopes()

        // Attached now, not when the EXISTS was built: the group is nested so
        // the correlated column condition stays AND-ed with the whole user
        // expression (an `orWhere` inside the callback must not swallow it),
        // and top-level AND order carries no meaning, so appending is safe.
        if (constraints.getConditions().length > 0) {
          conditions.where(constraints)
        }
      }
    }
  }

  // ── Conditional clauses ──────────────────────────────────────────────────────

  public when(condition: any, callback: (query: this) => void): this {
    if (condition) {
      callback(this)
    }

    return this
  }

  // ── Eager loading ────────────────────────────────────────────────────────────

  public with(relations: Record<string, (query: EntityQuery<any>) => void>): this
  public with(relations: string | string[], ...rest: string[]): this
  public with(
    relations: string | string[] | Record<string, (query: EntityQuery<any>) => void>,
    ...rest: string[]
  ): this {
    if (typeof relations === 'string') {
      this._withs.push({ path: relations }, ...rest.map((r) => ({ path: r })))
    } else if (Array.isArray(relations)) {
      this._withs.push(...relations.map((r) => ({ path: r })))
    } else {
      for (const [path, callback] of Object.entries(relations)) {
        this._withs.push({ path, callback })
      }
    }

    return this
  }

  // ── Join relationships ───────────────────────────────────────────────────────

  public joinRelationship(relationName: string): this {
    return this._applyJoin(JoinType.INNER_JOIN, relationName)
  }

  public innerJoinRelationship(relationName: string): this {
    return this._applyJoin(JoinType.INNER_JOIN, relationName)
  }

  public leftJoinRelationship(relationName: string): this {
    return this._applyJoin(JoinType.LEFT_JOIN, relationName)
  }

  // ── Existence checks ─────────────────────────────────────────────────────────

  public whereHas(relationName: string, callback?: (query: EntityQuery<any>) => void): this {
    return this._applyWhereHas('AND', relationName, callback)
  }

  public orWhereHas(relationName: string, callback?: (query: EntityQuery<any>) => void): this {
    return this._applyWhereHas('OR', relationName, callback)
  }

  private _applyWhereHas(
    connector: 'AND' | 'OR',
    relationName: string,
    callback?: (query: EntityQuery<any>) => void
  ): this {
    const rel = this._repository.relationships[relationName]

    if (!rel) {
      throw new Error(`Relationship "${relationName}" is not defined on ${this._repository.table}`)
    }

    const RelatedClass = rel.related()
    const relatedRepo = Repository.get(RelatedClass)
    const relatedTable = relatedRepo.table
    const source = this._repository.getSource()
    // Build a sub-DataTable and sub-EntityQuery so the callback can constrain
    // the related entity with proper field-name resolution.
    const subTable = source.table(relatedTable)
    const subEntityQuery = new EntityQuery(relatedRepo, subTable)

    if (callback) {
      callback(subEntityQuery)
    }

    // Build the top-level condition group for the subquery.
    //
    // The callback's own conditions are NOT folded in here: the related entity's
    // global scopes still have to be applied to `subEntityQuery`, and a scope
    // may be asynchronous while this method is a synchronous chainable. So the
    // sub-query is registered as pending and _resolvePendingRelationQueries
    // attaches its conditions later, from the terminal method — before the
    // statement is compiled, and wrapped in a nested group so the correlated
    // column condition stays AND-ed with the entire user expression (which
    // prevents SQL precedence bugs when the callback uses orWhere).
    const subConditions = new ConditionGroup()

    this._pendingRelationQueries.push({
      query: subEntityQuery,
      conditions: subConditions,
      constraints: subTable.getWhereConditions()
    })

    // Add the correlated join condition at the top level (always AND).
    if (rel.type === 'hasOne' || rel.type === 'hasMany') {
      subConditions.whereColumn(`${relatedTable}.${rel.foreignKey}`, `${this._repository.table}.${rel.localKey}`)
    } else if (rel.type === 'belongsTo') {
      subConditions.whereColumn(`${relatedTable}.${rel.localKey}`, `${this._repository.table}.${rel.foreignKey}`)
    } else if (rel.type === 'hasManyInArray') {
      // Correlated array membership: parent.arrayColumn @> ARRAY[related.localKey]
      subConditions.whereArrayContains(`${this._repository.table}.${rel.foreignKey}`, {
        name: rel.localKey,
        table: relatedTable
      })
    } else if (rel.type === 'hasOneThrough' || rel.type === 'hasManyThrough') {
      const ThroughClass = rel.through!()
      const throughRepo = Repository.get(ThroughClass)
      const throughTable = throughRepo.table

      // JOIN through table: related.foreign_key = through.through_local_key
      subTable.join(
        JoinType.INNER_JOIN,
        throughTable,
        `${relatedTable}.${rel.foreignKey}`,
        `${throughTable}.${rel.throughLocalKey}`
      )
      // Correlated condition: through.through_foreign_key = parent.local_key
      subConditions.whereColumn(`${throughTable}.${rel.throughForeignKey}`, `${this._repository.table}.${rel.localKey}`)
    } else if (rel.type === 'belongsToThrough') {
      const ThroughClass = rel.through!()
      const throughRepo = Repository.get(ThroughClass)
      const throughTable = throughRepo.table

      // JOIN through table: related.through_local_key = through.through_foreign_key
      subTable.join(
        JoinType.INNER_JOIN,
        throughTable,
        `${relatedTable}.${rel.throughLocalKey}`,
        `${throughTable}.${rel.throughForeignKey}`
      )
      // Correlated condition: through.local_key = self.foreign_key
      subConditions.whereColumn(`${throughTable}.${rel.localKey}`, `${this._repository.table}.${rel.foreignKey}`)
    }

    // Build EXISTS (SELECT 1 FROM related_table [...] WHERE [...]) via whereExists/orWhereExists.
    // Passing the SelectQuery instance (rather than a callback) keeps the subquery
    // compiled in the context of the outer statement — this ensures Postgres $N
    // positions are correct and avoids pre-baked placeholders clashing with outer bindings.
    const subSelectQuery = new SelectQuery(relatedTable)

    subSelectQuery.setSelectFields(['1'])
    subSelectQuery.setWhereConditions(subConditions)
    subSelectQuery.setJoins(subTable.getJoins())

    if (connector === 'AND') {
      this._table.whereExists(subSelectQuery)
    } else {
      this._table.orWhereExists(subSelectQuery)
    }

    return this
  }

  private _applyJoin(joinType: JoinType, relationName: string): this {
    const rel = this._repository.relationships[relationName]

    if (!rel) {
      throw new Error(`Relationship "${relationName}" is not defined on ${this._repository.table}`)
    }

    const RelatedClass = rel.related()
    const relatedRepo = Repository.get(RelatedClass)
    const relatedTable = relatedRepo.table

    if (rel.type === 'hasOne' || rel.type === 'hasMany') {
      const sourceField = `${this._repository.table}.${rel.localKey}`
      const remoteField = `${relatedTable}.${rel.foreignKey}`

      this._table.join(joinType, relatedTable, sourceField, remoteField)
    } else if (rel.type === 'belongsTo') {
      const sourceField = `${this._repository.table}.${rel.foreignKey}`
      const remoteField = `${relatedTable}.${rel.localKey}`

      this._table.join(joinType, relatedTable, sourceField, remoteField)
    } else if (rel.type === 'hasOneThrough' || rel.type === 'hasManyThrough') {
      const ThroughClass = rel.through!()
      const throughRepo = Repository.get(ThroughClass)
      const throughTable = throughRepo.table

      this._table.join(
        joinType,
        throughTable,
        `${this._repository.table}.${rel.localKey}`,
        `${throughTable}.${rel.throughForeignKey}`
      )
      this._table.join(
        joinType,
        relatedTable,
        `${throughTable}.${rel.throughLocalKey}`,
        `${relatedTable}.${rel.foreignKey}`
      )
    } else if (rel.type === 'belongsToThrough') {
      const ThroughClass = rel.through!()
      const throughRepo = Repository.get(ThroughClass)
      const throughTable = throughRepo.table

      // self.foreignKey → through.localKey
      this._table.join(
        joinType,
        throughTable,
        `${this._repository.table}.${rel.foreignKey}`,
        `${throughTable}.${rel.localKey}`
      )
      // through.throughForeignKey → related.throughLocalKey
      this._table.join(
        joinType,
        relatedTable,
        `${throughTable}.${rel.throughForeignKey}`,
        `${relatedTable}.${rel.throughLocalKey}`
      )
    }

    return this
  }

  // ── Terminal methods ─────────────────────────────────────────────────────────

  public async get(): Promise<T[]> {
    await this._applyGlobalScopes()
    const rows = await this._table.get()
    const entities = rows.map((row) => this._repository.fromRow(row))

    if (this._withs.length > 0) {
      await this._loadRelations(entities as any[], this._withs, this._repository)
    }

    return entities
  }

  public async first(): Promise<T | null> {
    await this._applyGlobalScopes()
    const row = await this._table.first()

    if (!row) {
      return null
    }

    const entity = this._repository.fromRow(row)

    if (this._withs.length > 0) {
      await this._loadRelations([entity as any], this._withs, this._repository)
    }

    return entity
  }

  public async count(column: Field = '*'): Promise<number> {
    await this._applyGlobalScopes()
    const resolved = column === '*' ? '*' : this._resolveField(column)

    return this._table.count(resolved)
  }

  public async paginate(perPage = 15, page = 1): Promise<PaginationResult<T>> {
    await this._applyGlobalScopes()
    const currentPage = Math.max(page, 1)
    const total = await this._table.count()
    const lastPage = Math.max(Math.ceil(total / perPage), 1)

    this._table.setOffset((currentPage - 1) * perPage).setLimit(perPage)
    // Reuse get() so entities are hydrated and any with() relations are loaded.
    const data = await this.get()
    const from = total === 0 ? null : (currentPage - 1) * perPage + 1
    const to = from === null ? null : from + data.length - 1

    return { data, total, perPage, currentPage, lastPage, from, to }
  }

  public async find(id: any): Promise<T | null> {
    await this._applyGlobalScopes()
    const row = await this._table.where(this._repository.primaryKey, id).first()

    if (!row) {
      return null
    }

    const entity = this._repository.fromRow(row)

    if (this._withs.length > 0) {
      await this._loadRelations([entity as any], this._withs, this._repository)
    }

    return entity
  }

  // ── Eager-load implementation ────────────────────────────────────────────────

  private async _loadRelations(entities: any[], withs: WithEntry[], parentRepo: EntityRepository<any>): Promise<void> {
    // Group by top-level relation name; carry along callback and nested paths
    const groups = new Map<string, { callback?: (query: EntityQuery<any>) => void; nested: WithEntry[] }>()

    for (const { path, callback } of withs) {
      const dot = path.indexOf('.')
      const head = dot === -1 ? path : path.substring(0, dot)
      const tail = dot === -1 ? null : path.substring(dot + 1)

      if (!groups.has(head)) {
        groups.set(head, { nested: [] })
      }

      const group = groups.get(head)!

      if (tail) {
        group.nested.push({ path: tail, callback })
      } else {
        group.callback = callback
      }
    }

    for (const [head, { callback, nested }] of groups) {
      const rel = parentRepo.relationships[head]

      if (!rel) {
        continue
      }

      const RelatedClass = rel.related()
      const relatedRepo = Repository.get(RelatedClass)
      let relatedItems: any[] = []

      if (rel.type === 'hasOne' || rel.type === 'hasMany') {
        const keys = [...new Set(entities.map((r) => r[rel.localKey]).filter((v) => v != null))]

        if (keys.length === 0) {
          continue
        }

        const q = relatedRepo.whereIn(rel.foreignKey, keys)

        if (callback) {
          callback(q)
        }

        relatedItems = await q.get()

        const lookup = new Map<any, any[]>()

        relatedItems.forEach((item: any) => {
          const k = item[rel.foreignKey]

          if (!lookup.has(k)) {
            lookup.set(k, [])
          }

          lookup.get(k)!.push(item)
        })

        entities.forEach((r) => {
          const matched = lookup.get(r[rel.localKey]) ?? []

          r[head] = rel.type === 'hasOne' ? matched[0] ?? null : matched
        })
      } else if (rel.type === 'belongsTo') {
        const keys = [...new Set(entities.map((r) => r[rel.foreignKey]).filter((v) => v != null))]

        if (keys.length === 0) {
          continue
        }

        const q = relatedRepo.whereIn(rel.localKey, keys)

        if (callback) {
          callback(q)
        }

        relatedItems = await q.get()

        const lookup = new Map<any, any>()

        relatedItems.forEach((item: any) => lookup.set(item[rel.localKey], item))

        entities.forEach((r) => {
          r[head] = lookup.get(r[rel.foreignKey]) ?? null
        })
      } else if (rel.type === 'hasManyInArray') {
        // The array of related keys lives on THIS (parent) side; gather their union.
        const keys = [
          ...new Set(
            entities
              .flatMap((r) => (Array.isArray(r[rel.foreignKey]) ? r[rel.foreignKey] : []))
              .filter((v: any) => v != null)
          )
        ]

        if (keys.length === 0) {
          entities.forEach((r) => {
            r[head] = []
          })

          continue
        }

        const q = relatedRepo.whereIn(rel.localKey, keys)

        if (callback) {
          callback(q)
        }

        relatedItems = await q.get()

        const lookup = new Map<any, any>()

        relatedItems.forEach((item: any) => lookup.set(item[rel.localKey], item))

        entities.forEach((r) => {
          const ids: any[] = Array.isArray(r[rel.foreignKey]) ? r[rel.foreignKey] : []

          // Preserve the array order; drop keys that did not resolve to a row.
          r[head] = ids.map((id: any) => lookup.get(id)).filter((v: any) => v != null)
        })
      } else if (rel.type === 'hasOneThrough' || rel.type === 'hasManyThrough') {
        const ThroughClass = rel.through!()
        const throughRepo = Repository.get(ThroughClass)
        const localKeys = [...new Set(entities.map((r) => r[rel.localKey]).filter((v) => v != null))]

        if (localKeys.length === 0) {
          continue
        }

        // Check if the through entities are already loaded on the parent entities
        // (e.g. via a prior .with('categories') when 'competitors' goes through TournamentCategory).
        // The through may be loaded as an array (HasMany) or a single object (HasOne/BelongsTo).
        // If so, reuse them to avoid a redundant DB round-trip.
        let throughItems: any[]
        const cachedThroughItems = entities.flatMap((r) => {
          for (const key of Object.keys(r)) {
            const val = r[key]

            if (Array.isArray(val) && val.length > 0 && val[0] instanceof ThroughClass) {
              return val
            }

            if (val != null && !Array.isArray(val) && val instanceof ThroughClass) {
              return [val]
            }
          }

          return []
        })

        if (cachedThroughItems.length > 0) {
          throughItems = cachedThroughItems
        } else {
          throughItems = await throughRepo.whereIn(rel.throughForeignKey!, localKeys).get()
        }

        const throughKeys = [
          ...new Set(throughItems.map((t: any) => t[rel.throughLocalKey!]).filter((v: any) => v != null))
        ]

        if (throughKeys.length === 0) {
          continue
        }

        const throughByParent = new Map<any, any[]>()

        throughItems.forEach((t: any) => {
          const k = t[rel.throughForeignKey!]

          if (!throughByParent.has(k)) {
            throughByParent.set(k, [])
          }

          throughByParent.get(k)!.push(t)
        })

        const q = relatedRepo.whereIn(rel.foreignKey, throughKeys)

        if (callback) {
          callback(q)
        }

        relatedItems = await q.get()

        const relatedByThrough = new Map<any, any[]>()

        relatedItems.forEach((item: any) => {
          const k = item[rel.foreignKey]

          if (!relatedByThrough.has(k)) {
            relatedByThrough.set(k, [])
          }

          relatedByThrough.get(k)!.push(item)
        })

        entities.forEach((r) => {
          const throughs = throughByParent.get(r[rel.localKey]) ?? []
          const matched: any[] = []

          throughs.forEach((t: any) => {
            const items = relatedByThrough.get(t[rel.throughLocalKey!]) ?? []

            matched.push(...items)
          })
          r[head] = rel.type === 'hasOneThrough' ? matched[0] ?? null : matched
        })
      } else if (rel.type === 'belongsToThrough') {
        const ThroughClass = rel.through!()
        const throughRepo = Repository.get(ThroughClass)
        // Step 1: load through items matching self.foreignKey → through.localKey
        const foreignKeys = [...new Set(entities.map((r) => r[rel.foreignKey]).filter((v) => v != null))]

        if (foreignKeys.length === 0) {
          continue
        }

        // Check if the through entities are already loaded as single-object properties
        // (e.g. via a prior .with('someRelation') where the related model is ThroughClass).
        const cachedThroughItems = entities.flatMap((r) => {
          for (const key of Object.keys(r)) {
            const val = r[key]
            if (val != null && !Array.isArray(val) && val instanceof ThroughClass) {
              return [val]
            }
          }
          return []
        })

        const throughItems =
          cachedThroughItems.length > 0
            ? cachedThroughItems
            : await throughRepo.whereIn(rel.localKey, foreignKeys).get()
        // Step 2: load related items matching through.throughForeignKey → related.throughLocalKey
        const throughFKValues = [
          ...new Set(throughItems.map((t: any) => t[rel.throughForeignKey!]).filter((v: any) => v != null))
        ]

        if (throughFKValues.length === 0) {
          continue
        }

        const q = relatedRepo.whereIn(rel.throughLocalKey, throughFKValues)

        if (callback) {
          callback(q)
        }

        relatedItems = await q.get()

        // Build lookup: through.localKey → through item
        const throughByLocalKey = new Map<any, any>()

        throughItems.forEach((t: any) => throughByLocalKey.set(t[rel.localKey], t))

        // Build lookup: related.throughLocalKey → related item
        const relatedByThroughLocalKey = new Map<any, any>()

        relatedItems.forEach((item: any) => relatedByThroughLocalKey.set(item[rel.throughLocalKey!], item))

        // Assign: entity → through → related
        entities.forEach((r) => {
          const throughItem = throughByLocalKey.get(r[rel.foreignKey])
          const relatedItem = throughItem ? relatedByThroughLocalKey.get(throughItem[rel.throughForeignKey!]) : null

          r[head] = relatedItem ?? null
        })
      }

      // Recurse for nested dot-notation paths
      if (nested.length > 0 && relatedItems.length > 0) {
        await this._loadRelations(relatedItems, nested, relatedRepo)
      }
    }
  }

  // ── Field-name resolution ────────────────────────────────────────────────────

  private _resolveFieldName(name: string): string {
    const columnsMap = this._repository.columnsMap

    return columnsMap[name] ?? name
  }

  private _resolveField<F extends Field>(field: F): F {
    if (typeof field === 'string') {
      const dot = field.lastIndexOf('.')

      if (dot === -1) {
        return this._resolveFieldName(field) as F
      }

      const tablePart = field.substring(0, dot)

      if (tablePart !== this._repository.table) {
        return field
      }

      return `${tablePart}.${this._resolveFieldName(field.substring(dot + 1))}` as F
    }

    if (field && typeof field === 'object' && typeof (field as any).name === 'string') {
      const table = (field as any).table
      const tableName = typeof table === 'string' ? table : table?.name

      if (tableName != null && tableName !== this._repository.table) {
        return field
      }

      return { ...(field as any), name: this._resolveFieldName((field as any).name) }
    }

    return field
  }

  private _resolveCondition(condition: any): any {
    if (typeof condition === 'function') {
      return (group: ConditionGroup) => {
        condition(group)
        this._resolveConditionGroup(group)
      }
    }

    if (condition instanceof ConditionGroup) {
      this._resolveConditionGroup(condition)

      return condition
    }

    if (condition && typeof condition === 'object' && 'field' in condition) {
      const resolved: any = { ...condition, field: this._resolveField(condition.field) }

      if ('column' in condition) {
        resolved.column = this._resolveField(condition.column)
      }

      return resolved
    }

    return condition
  }

  private _resolveConditionGroup(group: ConditionGroup): void {
    for (const entry of group.getConditions()) {
      entry.condition = this._resolveCondition(entry.condition)
    }
  }

  // ── DataTable method proxies ─────────────────────────────────────────────────

  public where(callback: (group: ConditionGroup) => void): this
  public where(condition: Condition): this
  public where(field: Field, value: any): this
  public where(field: Field, operator: string, value: any): this
  public where(...args: any[]): this {
    if (args.length >= 2) {
      args[0] = this._resolveField(args[0])
    } else if (args.length === 1) {
      args[0] = this._resolveCondition(args[0])
    }

    ;(this._table as any).where(...args)

    return this
  }

  public whereIn(field: Field, values: any[]): this {
    this._table.whereIn(this._resolveField(field), values)

    return this
  }

  public whereNotIn(field: Field, values: any[]): this {
    this._table.whereNotIn(this._resolveField(field), values)

    return this
  }

  public whereBetween(field: Field, range: [any, any]): this {
    this._table.whereBetween(this._resolveField(field), range)

    return this
  }

  public whereNotBetween(field: Field, range: [any, any]): this {
    this._table.whereNotBetween(this._resolveField(field), range)

    return this
  }

  public whereNull(field: Field): this {
    this._table.whereNull(this._resolveField(field))

    return this
  }

  public whereNotNull(field: Field): this {
    this._table.whereNotNull(this._resolveField(field))

    return this
  }

  public whereLike(field: Field, pattern: string, caseSensitive = false): this {
    this._table.whereLike(this._resolveField(field), pattern, caseSensitive)

    return this
  }

  public whereNotLike(field: Field, pattern: string, caseSensitive = false): this {
    this._table.whereNotLike(this._resolveField(field), pattern, caseSensitive)

    return this
  }

  public whereColumn(field: Field, column: Field): this
  public whereColumn(field: Field, operator: string, column: Field): this
  public whereColumn(...args: any[]): this {
    args[0] = this._resolveField(args[0])
    args[args.length - 1] = this._resolveField(args[args.length - 1])
    ;(this._table as any).whereColumn(...args)

    return this
  }

  public whereExists(subquery: ExistsSubquery): this {
    this._table.whereExists(subquery)

    return this
  }

  public whereNotExists(subquery: ExistsSubquery): this {
    this._table.whereNotExists(subquery)

    return this
  }

  public whereArrayContains(field: Field, value: any): this {
    this._table.whereArrayContains(this._resolveField(field), value)

    return this
  }

  public orWhere(...args: any[]): this {
    if (args.length >= 2) {
      args[0] = this._resolveField(args[0])
    } else if (args.length === 1) {
      args[0] = this._resolveCondition(args[0])
    }

    ;(this._table as any).orWhere(...args)

    return this
  }

  public orWhereIn(field: Field, values: any[]): this {
    this._table.orWhereIn(this._resolveField(field), values)

    return this
  }

  public orWhereNotIn(field: Field, values: any[]): this {
    this._table.orWhereNotIn(this._resolveField(field), values)

    return this
  }

  public orWhereBetween(field: Field, range: [any, any]): this {
    this._table.orWhereBetween(this._resolveField(field), range)

    return this
  }

  public orWhereNotBetween(field: Field, range: [any, any]): this {
    this._table.orWhereNotBetween(this._resolveField(field), range)

    return this
  }

  public orWhereNull(field: Field): this {
    this._table.orWhereNull(this._resolveField(field))

    return this
  }

  public orWhereNotNull(field: Field): this {
    this._table.orWhereNotNull(this._resolveField(field))

    return this
  }

  public orWhereLike(field: Field, pattern: string, caseSensitive = false): this {
    this._table.orWhereLike(this._resolveField(field), pattern, caseSensitive)

    return this
  }

  public orWhereNotLike(field: Field, pattern: string, caseSensitive = false): this {
    this._table.orWhereNotLike(this._resolveField(field), pattern, caseSensitive)

    return this
  }

  public orWhereExists(subquery: ExistsSubquery): this {
    this._table.orWhereExists(subquery)

    return this
  }

  public orWhereNotExists(subquery: ExistsSubquery): this {
    this._table.orWhereNotExists(subquery)

    return this
  }

  public orWhereArrayContains(field: Field, value: any): this {
    this._table.orWhereArrayContains(this._resolveField(field), value)

    return this
  }

  public select(...fields: (Field | Field[])[]): this {
    const resolved = fields.map((f) =>
      Array.isArray(f) ? f.map((inner) => this._resolveField(inner)) : this._resolveField(f)
    )

    ;(this._table as any).select(...resolved)

    return this
  }

  public orderBy(field: Field, direction?: OrderByDirection): this {
    this._table.orderBy(this._resolveField(field), direction as any)

    return this
  }

  public orderByDesc(field: Field): this {
    return this.orderBy(field, OrderByDirection.DESC)
  }

  public groupBy(...fields: Field[]): this {
    ;(this._table as any).groupBy(...fields.map((f) => this._resolveField(f)))

    return this
  }

  public limit(value: number): this {
    this._table.setLimit(value)

    return this
  }

  public offset(value: number): this {
    this._table.setOffset(value)

    return this
  }

  public distinct(): this {
    this._table.setDistinct(true)

    return this
  }
}
