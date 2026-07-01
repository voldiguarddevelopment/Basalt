// The symbol table: a stack of scopes (global -> namespace -> function -> block, pushed and
// popped in lockstep with the AST nesting the checker walks). Each scope keeps four
// independent namespaces, matching real C: ordinary identifiers (vars/functions/enum
// constants share one namespace), struct tags, union tags, and typedef names. Enum tags get
// their own namespace too, tracked separately from enum constants.
//
// Struct/union *field* names are not part of this table at all — member-access checking looks
// them up directly on the `StructInfo` for the accessed type.

use std::collections::HashMap;

use crate::ty::Ty;

#[derive(Debug, Clone)]
pub(crate) struct StructInfo {
    pub fields: Vec<(String, Ty)>,
}

#[derive(Debug, Clone)]
pub(crate) struct FuncSig {
    pub ret: Ty,
    pub params: Vec<Ty>,
    pub variadic: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum ValueSym {
    Var(Ty),
    Func(FuncSig),
    EnumConst(Ty),
}

#[derive(Debug, Default)]
pub(crate) struct Scope {
    pub values: HashMap<String, ValueSym>,
    pub structs: HashMap<String, StructInfo>,
    pub unions: HashMap<String, StructInfo>,
    pub enums: HashMap<String, ()>,
    pub typedefs: HashMap<String, Ty>,
}

/// The scope stack. Innermost scope is the last element; lookups walk from the back forward.
#[derive(Debug, Default)]
pub(crate) struct ScopeStack {
    scopes: Vec<Scope>,
}

impl ScopeStack {
    pub fn new() -> ScopeStack {
        ScopeStack { scopes: Vec::new() }
    }

    pub fn push(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub fn pop(&mut self) {
        self.scopes.pop();
    }

    fn top_mut(&mut self) -> &mut Scope {
        self.scopes
            .last_mut()
            .expect("scope stack must not be empty while checking")
    }

    /// Inserts into the innermost scope's value namespace. Returns `true` if a symbol with
    /// this name already existed in that same scope (the caller reports `E302`) — shadowing
    /// an outer scope's binding is not a redefinition, only colliding in the same scope is.
    pub fn declare_value(&mut self, name: &str, sym: ValueSym) -> bool {
        let existed = self.top_mut().values.contains_key(name);
        self.top_mut().values.insert(name.to_string(), sym);
        existed
    }

    pub fn declare_struct(&mut self, name: &str, info: StructInfo) -> bool {
        let existed = self.top_mut().structs.contains_key(name);
        self.top_mut().structs.insert(name.to_string(), info);
        existed
    }

    pub fn declare_union(&mut self, name: &str, info: StructInfo) -> bool {
        let existed = self.top_mut().unions.contains_key(name);
        self.top_mut().unions.insert(name.to_string(), info);
        existed
    }

    pub fn declare_enum(&mut self, name: &str) -> bool {
        let existed = self.top_mut().enums.contains_key(name);
        self.top_mut().enums.insert(name.to_string(), ());
        existed
    }

    pub fn declare_typedef(&mut self, name: &str, ty: Ty) -> bool {
        let existed = self.top_mut().typedefs.contains_key(name);
        self.top_mut().typedefs.insert(name.to_string(), ty);
        existed
    }

    pub fn lookup_value(&self, name: &str) -> Option<&ValueSym> {
        self.scopes.iter().rev().find_map(|s| s.values.get(name))
    }

    pub fn lookup_struct(&self, name: &str) -> Option<&StructInfo> {
        self.scopes.iter().rev().find_map(|s| s.structs.get(name))
    }

    pub fn lookup_union(&self, name: &str) -> Option<&StructInfo> {
        self.scopes.iter().rev().find_map(|s| s.unions.get(name))
    }

    pub fn lookup_enum(&self, name: &str) -> Option<()> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.enums.get(name).copied())
    }

    pub fn lookup_typedef(&self, name: &str) -> Option<&Ty> {
        self.scopes.iter().rev().find_map(|s| s.typedefs.get(name))
    }
}
