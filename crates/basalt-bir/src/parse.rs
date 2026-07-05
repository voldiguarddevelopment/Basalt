// BIR textual parser: hand-rolled lexer + recursive-descent parser, std-only, no
// external crates (this project is hand-rolled by identity). Reads exactly the
// grammar documented in `lib.rs`'s crate header, which `print.rs` emits. The two files
// mirror each other opcode-by-opcode; see the note at the top of `print.rs`.
//
// Value and block numbering doubles as an ordering check: a `%<n> =` prefix or `bb<n>:`
// label must name the arena index the parser is about to assign (params-then-instructions
// order, matching `print.rs`'s output), or parsing fails. This is what lets a plain `Vec`
// push reconstruct the exact same arena as the one that was printed.

use std::fmt;

use crate::ir::{
    AtomicOp, BinOp, Block, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId,
    LaunchBounds, MmaLayout, Module, Op, ShuffleKind, Term, ValRef,
};
use crate::ty::{AddrSpace, Scalar, Ty};

/// A BIR textual-parse failure: the 1-based source line plus a description. Local to this
/// crate — callers that need a language-neutral E-code (`basalt-diag`) wrap this
/// at the boundary rather than this crate depending on that registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parses BIR textual form into a `Module`. `parse(print(m)) == m` is the round-trip
/// invariant this crate's tests hold (`tests/roundtrip.rs`).
pub fn parse(src: &str) -> Result<Module, ParseError> {
    let toks = lex(src);
    let mut p = Parser {
        toks: &toks,
        pos: 0,
    };
    let m = p.parse_module()?;
    p.expect_tok(Tok::Eof)?;
    Ok(m)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Eq,
    Arrow,
    Eof,
}

struct Token {
    tok: Tok,
    line: usize,
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '@' | '%' | '-')
}

fn lex(src: &str) -> Vec<Token> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_whitespace() => i += 1,
            '(' => {
                toks.push(Token {
                    tok: Tok::LParen,
                    line,
                });
                i += 1;
            }
            ')' => {
                toks.push(Token {
                    tok: Tok::RParen,
                    line,
                });
                i += 1;
            }
            '{' => {
                toks.push(Token {
                    tok: Tok::LBrace,
                    line,
                });
                i += 1;
            }
            '}' => {
                toks.push(Token {
                    tok: Tok::RBrace,
                    line,
                });
                i += 1;
            }
            '[' => {
                toks.push(Token {
                    tok: Tok::LBracket,
                    line,
                });
                i += 1;
            }
            ']' => {
                toks.push(Token {
                    tok: Tok::RBracket,
                    line,
                });
                i += 1;
            }
            ',' => {
                toks.push(Token {
                    tok: Tok::Comma,
                    line,
                });
                i += 1;
            }
            ':' => {
                toks.push(Token {
                    tok: Tok::Colon,
                    line,
                });
                i += 1;
            }
            '=' => {
                toks.push(Token { tok: Tok::Eq, line });
                i += 1;
            }
            '-' if chars.get(i + 1) == Some(&'>') => {
                toks.push(Token {
                    tok: Tok::Arrow,
                    line,
                });
                i += 2;
            }
            _ => {
                let start = i;
                while i < chars.len() && is_word_char(chars[i]) {
                    i += 1;
                }
                if i == start {
                    // Unrecognized punctuation: skip it and let the parser report the
                    // resulting malformed structure with useful context.
                    i += 1;
                } else {
                    let w: String = chars[start..i].iter().collect();
                    toks.push(Token {
                        tok: Tok::Word(w),
                        line,
                    });
                }
            }
        }
    }
    let last_line = toks.last().map_or(line, |t| t.line);
    toks.push(Token {
        tok: Tok::Eof,
        line: last_line,
    });
    toks
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn cur(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn line(&self) -> usize {
        self.cur().line
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            line: self.line(),
            msg: msg.into(),
        }
    }

    fn bump(&mut self) -> &Token {
        let t = &self.toks[self.pos];
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn check(&self, tok: &Tok) -> bool {
        &self.cur().tok == tok
    }

    fn expect_tok(&mut self, tok: Tok) -> Result<(), ParseError> {
        if self.cur().tok == tok {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected {:?}, found {:?}", tok, self.cur().tok)))
        }
    }

    fn peek_word(&self) -> Option<&str> {
        match &self.cur().tok {
            Tok::Word(w) => Some(w.as_str()),
            _ => None,
        }
    }

    fn word(&mut self) -> Result<String, ParseError> {
        match &self.cur().tok {
            Tok::Word(w) => {
                let w = w.clone();
                self.bump();
                Ok(w)
            }
            other => Err(self.err(format!("expected a word, found {other:?}"))),
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<(), ParseError> {
        match self.peek_word() {
            Some(w) if w == kw => {
                self.bump();
                Ok(())
            }
            _ => Err(self.err(format!("expected `{kw}`, found {:?}", self.cur().tok))),
        }
    }

    fn ty(&mut self) -> Result<Ty, ParseError> {
        let w = self.word()?;
        Ty::parse(&w).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("unknown type `{w}`"),
        })
    }

    fn val(&mut self) -> Result<ValRef, ParseError> {
        let w = self.word()?;
        val_from_word(&w).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("expected a value reference, found `{w}`"),
        })
    }

    fn block_id(&mut self) -> Result<BlockId, ParseError> {
        let w = self.word()?;
        block_id_from_word(&w).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("expected a block label, found `{w}`"),
        })
    }

    fn scalar_ty(&mut self) -> Result<Scalar, ParseError> {
        let w = self.word()?;
        Scalar::parse(&w).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("unknown scalar type `{w}`"),
        })
    }

    fn mma_layout(&mut self) -> Result<MmaLayout, ParseError> {
        let w = self.word()?;
        MmaLayout::parse(&w).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("unknown mma layout `{w}`"),
        })
    }

    fn u32_lit(&mut self) -> Result<u32, ParseError> {
        let w = self.word()?;
        w.parse::<u32>().map_err(|_| ParseError {
            line: self.line(),
            msg: format!("expected an integer, found `{w}`"),
        })
    }

    fn i64_lit(&mut self) -> Result<i64, ParseError> {
        let w = self.word()?;
        w.parse::<i64>().map_err(|_| ParseError {
            line: self.line(),
            msg: format!("expected an integer, found `{w}`"),
        })
    }

    fn f64_lit(&mut self) -> Result<f64, ParseError> {
        let w = self.word()?;
        w.parse::<f64>().map_err(|_| ParseError {
            line: self.line(),
            msg: format!("expected a float, found `{w}`"),
        })
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        self.expect_kw("module")?;
        self.expect_tok(Tok::LBrace)?;

        let mut launch_bounds = None;
        if self.peek_word() == Some("launch_bounds") {
            self.bump();
            self.expect_attr_key("max_threads")?;
            let max_threads = self.u32_lit()?;
            self.expect_attr_key("min_blocks")?;
            let min_blocks = self.u32_lit()?;
            launch_bounds = Some(LaunchBounds {
                max_threads,
                min_blocks,
            });
        }

        self.expect_kw("shared_mem_bytes")?;
        let shared_mem_bytes = self.u32_lit()?;

        self.expect_kw("target_dtypes")?;
        let mut target_dtypes = Vec::new();
        while let Some(w) = self.peek_word() {
            match Scalar::parse(w) {
                Some(s) => {
                    target_dtypes.push(s);
                    self.bump();
                }
                None => break,
            }
        }

        let mut funcs = Vec::new();
        while matches!(self.peek_word(), Some("func") | Some("host")) {
            funcs.push(self.parse_func()?);
        }

        self.expect_tok(Tok::RBrace)?;
        Ok(Module {
            funcs,
            launch_bounds,
            shared_mem_bytes,
            target_dtypes,
        })
    }

    /// `max_threads=` / `min_blocks=` — a bare word immediately followed by `=`, both
    /// consumed here since the lexer doesn't glue `key=value` into one token.
    fn expect_attr_key(&mut self, key: &str) -> Result<(), ParseError> {
        self.expect_kw(key)?;
        self.expect_tok(Tok::Eq)
    }

    /// A function's own `is_kernel` marker is spelled as an optional `host` keyword
    /// immediately before `func`: bare `func` means `is_kernel = true` (every function BIR has
    /// ever printed before this field existed was a kernel, so this default keeps every prior
    /// `.bir` fixture parsing unchanged), `host func` means `is_kernel = false`.
    fn parse_func(&mut self) -> Result<Function, ParseError> {
        let is_kernel = if self.peek_word() == Some("host") {
            self.bump();
            false
        } else {
            true
        };
        self.expect_kw("func")?;
        let name_word = self.word()?;
        let name = name_word
            .strip_prefix('@')
            .ok_or_else(|| ParseError {
                line: self.line(),
                msg: format!("expected `@name`, found `{name_word}`"),
            })?
            .to_string();

        self.expect_tok(Tok::LParen)?;
        let mut params = Vec::new();
        if !self.check(&Tok::RParen) {
            loop {
                params.push(self.ty()?);
                if self.check(&Tok::Comma) {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        self.expect_tok(Tok::RParen)?;
        self.expect_tok(Tok::Arrow)?;
        let ret = self.ty()?;
        self.expect_tok(Tok::LBrace)?;

        let mut blocks = Vec::new();
        let mut insts = Vec::new();
        while !self.check(&Tok::RBrace) {
            self.parse_block(&mut blocks, &mut insts)?;
        }
        self.expect_tok(Tok::RBrace)?;

        Ok(Function {
            name,
            is_kernel,
            params,
            ret,
            blocks,
            insts,
        })
    }

    fn parse_block(
        &mut self,
        blocks: &mut Vec<Block>,
        insts: &mut Vec<Inst>,
    ) -> Result<(), ParseError> {
        let label = self.word()?;
        let n = block_id_from_word(&label).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("expected a block label, found `{label}`"),
        })?;
        let expected = blocks.len() as u32;
        if n.0 != expected {
            return Err(self.err(format!(
                "block label bb{} out of order, expected bb{expected}",
                n.0
            )));
        }
        self.expect_tok(Tok::Colon)?;

        let mut ids = Vec::new();
        loop {
            match self.peek_word() {
                Some("br") | Some("condbr") | Some("switch") | Some("ret") => break,
                None => return Err(self.err("unexpected end of input inside a block")),
                _ => {}
            }
            let id = insts.len() as u32;
            let inst = self.parse_inst(id)?;
            insts.push(inst);
            ids.push(InstId(id));
        }
        let term = self.parse_term()?;
        blocks.push(Block { insts: ids, term });
        Ok(())
    }

    /// Parses one instruction. `expected_id` is the arena index it must land at (the
    /// instruction arena is only ever appended to, in print order).
    fn parse_inst(&mut self, expected_id: u32) -> Result<Inst, ParseError> {
        let has_prefix = matches!(self.peek_word(), Some(w) if w.starts_with('%'))
            && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Eq));
        if has_prefix {
            let w = self.word()?;
            let n: u32 = w
                .strip_prefix('%')
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| ParseError {
                    line: self.line(),
                    msg: format!("bad value name `{w}`"),
                })?;
            if n != expected_id {
                return Err(self.err(format!("value %{n} out of order, expected %{expected_id}")));
            }
            self.expect_tok(Tok::Eq)?;
        }

        let op_word = self.word()?;
        let (ty, op) = self.parse_op(&op_word)?;
        let inst = Inst { ty, op };

        if has_prefix != inst.has_result() {
            return Err(self.err(format!(
                "`{op_word}` {} a result but {}",
                if inst.has_result() {
                    "produces"
                } else {
                    "does not produce"
                },
                if has_prefix {
                    "one was named"
                } else {
                    "none was named"
                },
            )));
        }
        Ok(inst)
    }

    fn parse_op(&mut self, op_word: &str) -> Result<(Ty, Op), ParseError> {
        if let Some(b) = BinOp::parse(op_word) {
            let ty = self.ty()?;
            let a = self.val()?;
            self.expect_tok(Tok::Comma)?;
            let bb = self.val()?;
            return Ok((ty, Op::Bin(b, a, bb)));
        }
        if let Some(c) = CastOp::parse(op_word) {
            let dst = self.ty()?;
            let src = self.ty()?;
            let v = self.val()?;
            return Ok((dst, Op::Cast(c, src, v)));
        }
        if let Some(k) = ShuffleKind::parse(op_word) {
            let ty = self.ty()?;
            let a = self.val()?;
            self.expect_tok(Tok::Comma)?;
            let b = self.val()?;
            return Ok((ty, Op::Shuffle(k, a, b)));
        }
        if let Some(a) = AtomicOp::parse(op_word) {
            let ty = self.ty()?;
            let space = self.ptr_space()?;
            let ptr = self.val()?;
            self.expect_tok(Tok::Comma)?;
            let v = self.val()?;
            return Ok((ty, Op::Atomic(a, ptr, v, space)));
        }
        if let Some(op) = Op::gpu_index_from_text(op_word) {
            let ty = self.ty()?;
            return Ok((ty, op));
        }

        match op_word {
            "const.i" => {
                let ty = self.ty()?;
                let v = self.i64_lit()?;
                Ok((ty, Op::ConstInt(v)))
            }
            "const.f" => {
                let ty = self.ty()?;
                let v = self.f64_lit()?;
                Ok((ty, Op::ConstFloat(v)))
            }
            "icmp" => {
                let pred_word = self.word()?;
                let pred = ICmpPred::parse(&pred_word).ok_or_else(|| ParseError {
                    line: self.line(),
                    msg: format!("unknown icmp predicate `{pred_word}`"),
                })?;
                let oty = self.ty()?;
                let a = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b = self.val()?;
                Ok((Ty::Scalar(Scalar::I1), Op::ICmp(pred, oty, a, b)))
            }
            "fcmp" => {
                let pred_word = self.word()?;
                let pred = FCmpPred::parse(&pred_word).ok_or_else(|| ParseError {
                    line: self.line(),
                    msg: format!("unknown fcmp predicate `{pred_word}`"),
                })?;
                let oty = self.ty()?;
                let a = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b = self.val()?;
                Ok((Ty::Scalar(Scalar::I1), Op::FCmp(pred, oty, a, b)))
            }
            "select" => {
                let ty = self.ty()?;
                let c = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let a = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b = self.val()?;
                Ok((ty, Op::Select(c, a, b)))
            }
            "load" => {
                let ty = self.ty()?;
                let space = self.ptr_space()?;
                let ptr = self.val()?;
                self.expect_tok(Tok::Comma)?;
                self.expect_kw("align")?;
                let align = self.u32_lit()?;
                let volatile = self.eat_volatile()?;
                Ok((
                    ty,
                    Op::Load {
                        ptr,
                        space,
                        align,
                        volatile,
                    },
                ))
            }
            "store" => {
                let ty = self.ty()?;
                let space = self.ptr_space()?;
                let ptr = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let v = self.val()?;
                self.expect_tok(Tok::Comma)?;
                self.expect_kw("align")?;
                let align = self.u32_lit()?;
                let volatile = self.eat_volatile()?;
                Ok((
                    Ty::Void,
                    Op::Store {
                        ptr,
                        val: v,
                        ty,
                        space,
                        align,
                        volatile,
                    },
                ))
            }
            "phi" => {
                let ty = self.ty()?;
                self.expect_tok(Tok::LBracket)?;
                let mut incoming = Vec::new();
                if !self.check(&Tok::RBracket) {
                    loop {
                        let bb = self.block_id()?;
                        self.expect_tok(Tok::Arrow)?;
                        let v = self.val()?;
                        incoming.push((bb, v));
                        if self.check(&Tok::Comma) {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect_tok(Tok::RBracket)?;
                Ok((ty, Op::Phi(incoming)))
            }
            "barrier" => Ok((Ty::Void, Op::Barrier)),
            "ballot" => {
                let ty = self.ty()?;
                let a = self.val()?;
                Ok((ty, Op::Ballot(a)))
            }
            "vote.any" => {
                let ty = self.ty()?;
                let a = self.val()?;
                Ok((ty, Op::VoteAny(a)))
            }
            "vote.all" => {
                let ty = self.ty()?;
                let a = self.val()?;
                Ok((ty, Op::VoteAll(a)))
            }
            "atomic.cas" => {
                let ty = self.ty()?;
                let space = self.ptr_space()?;
                let ptr = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let cmp = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let new = self.val()?;
                Ok((ty, Op::AtomicCas(ptr, cmp, new, space)))
            }
            "kernel.launch" => {
                let name_word = self.word()?;
                let kernel = name_word
                    .strip_prefix('@')
                    .ok_or_else(|| ParseError {
                        line: self.line(),
                        msg: format!("expected `@name`, found `{name_word}`"),
                    })?
                    .to_string();
                self.expect_kw("grid")?;
                let g0 = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let g1 = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let g2 = self.val()?;
                self.expect_kw("block")?;
                let b0 = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b1 = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b2 = self.val()?;
                self.expect_kw("shared")?;
                let shared = self.val()?;
                self.expect_kw("stream")?;
                let stream = self.val()?;
                self.expect_tok(Tok::LBracket)?;
                let mut args = Vec::new();
                if !self.check(&Tok::RBracket) {
                    loop {
                        args.push(self.val()?);
                        if self.check(&Tok::Comma) {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect_tok(Tok::RBracket)?;
                Ok((
                    Ty::Void,
                    Op::KernelLaunch {
                        kernel,
                        grid: [g0, g1, g2],
                        block: [b0, b1, b2],
                        shared,
                        stream,
                        args,
                    },
                ))
            }
            "cuda.malloc" => {
                let ty = self.ty()?;
                let size = self.val()?;
                Ok((ty, Op::CudaMalloc { size }))
            }
            "cuda.memcpy" => {
                let dst = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let src = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let count = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let kind = self.val()?;
                Ok((
                    Ty::Void,
                    Op::CudaMemcpy {
                        dst,
                        src,
                        count,
                        kind,
                    },
                ))
            }
            "cuda.free" => {
                let ptr = self.val()?;
                Ok((Ty::Void, Op::CudaFree { ptr }))
            }
            "cuda.device_sync" => Ok((Ty::Void, Op::CudaDeviceSynchronize)),
            "call" => {
                let ty = self.ty()?;
                let name_word = self.word()?;
                let func = name_word
                    .strip_prefix('@')
                    .ok_or_else(|| ParseError {
                        line: self.line(),
                        msg: format!("expected `@name`, found `{name_word}`"),
                    })?
                    .to_string();
                self.expect_tok(Tok::LBracket)?;
                let mut args = Vec::new();
                if !self.check(&Tok::RBracket) {
                    loop {
                        args.push(self.val()?);
                        if self.check(&Tok::Comma) {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect_tok(Tok::RBracket)?;
                Ok((ty, Op::Call { func, args }))
            }
            "mma" => {
                let in_dtype = self.scalar_ty()?;
                let acc_dtype = self.scalar_ty()?;
                let layout_a = self.mma_layout()?;
                let layout_b = self.mma_layout()?;
                self.expect_kw("m")?;
                let m = self.u32_lit()?;
                self.expect_kw("n")?;
                let n = self.u32_lit()?;
                self.expect_kw("k")?;
                let k = self.u32_lit()?;
                let a = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let b = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let c = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let d = self.val()?;
                Ok((
                    Ty::Void,
                    Op::Mma {
                        a,
                        b,
                        c,
                        d,
                        m,
                        n,
                        k,
                        in_dtype,
                        acc_dtype,
                        layout_a,
                        layout_b,
                    },
                ))
            }
            other => Err(self.err(format!("unknown opcode `{other}`"))),
        }
    }

    fn ptr_space(&mut self) -> Result<AddrSpace, ParseError> {
        let w = self.word()?;
        let rest = w.strip_prefix("ptr.").ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("expected a pointer type, found `{w}`"),
        })?;
        AddrSpace::parse(rest).ok_or_else(|| ParseError {
            line: self.line(),
            msg: format!("unknown address space `{rest}`"),
        })
    }

    fn eat_volatile(&mut self) -> Result<bool, ParseError> {
        if self.check(&Tok::Comma) {
            let save = self.pos;
            self.bump();
            if self.peek_word() == Some("volatile") {
                self.bump();
                return Ok(true);
            }
            self.pos = save;
        }
        Ok(false)
    }

    fn parse_term(&mut self) -> Result<Term, ParseError> {
        let kw = self.word()?;
        match kw.as_str() {
            "br" => {
                let b = self.block_id()?;
                Ok(Term::Br(b))
            }
            "condbr" => {
                let c = self.val()?;
                self.expect_tok(Tok::Comma)?;
                let t = self.block_id()?;
                self.expect_tok(Tok::Comma)?;
                let f = self.block_id()?;
                Ok(Term::CondBr(c, t, f))
            }
            "switch" => {
                let v = self.val()?;
                self.expect_tok(Tok::Comma)?;
                self.expect_kw("default")?;
                let default = self.block_id()?;
                self.expect_tok(Tok::LBracket)?;
                let mut cases = Vec::new();
                if !self.check(&Tok::RBracket) {
                    loop {
                        let case = self.i64_lit()?;
                        self.expect_tok(Tok::Arrow)?;
                        let bb = self.block_id()?;
                        cases.push((case, bb));
                        if self.check(&Tok::Comma) {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect_tok(Tok::RBracket)?;
                Ok(Term::Switch(v, default, cases))
            }
            "ret" => {
                if matches!(self.peek_word(), Some(w) if w.starts_with('%')) {
                    Ok(Term::Ret(Some(self.val()?)))
                } else {
                    Ok(Term::Ret(None))
                }
            }
            other => Err(self.err(format!("expected a terminator, found `{other}`"))),
        }
    }
}

fn val_from_word(w: &str) -> Option<ValRef> {
    let body = w.strip_prefix('%')?;
    if let Some(n) = body.strip_prefix("arg") {
        n.parse().ok().map(ValRef::Param)
    } else {
        body.parse().ok().map(|n| ValRef::Val(InstId(n)))
    }
}

fn block_id_from_word(w: &str) -> Option<BlockId> {
    w.strip_prefix("bb")
        .and_then(|s| s.parse().ok())
        .map(BlockId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_module() {
        let src = "module {\n  shared_mem_bytes 0\n  target_dtypes\n}\n";
        let m = parse(src).unwrap();
        assert!(m.funcs.is_empty());
        assert_eq!(m.shared_mem_bytes, 0);
        assert_eq!(m.launch_bounds, None);
    }

    #[test]
    fn parses_launch_bounds_and_dtypes() {
        let src = "module {\n  launch_bounds max_threads=256 min_blocks=2\n  shared_mem_bytes 4096\n  target_dtypes i32 f32\n}\n";
        let m = parse(src).unwrap();
        assert_eq!(
            m.launch_bounds,
            Some(LaunchBounds {
                max_threads: 256,
                min_blocks: 2
            })
        );
        assert_eq!(m.shared_mem_bytes, 4096);
        assert_eq!(m.target_dtypes, vec![Scalar::I32, Scalar::F32]);
    }

    #[test]
    fn rejects_out_of_order_block_label() {
        let src = "module {\n  shared_mem_bytes 0\n  target_dtypes\n\n  func @f() -> void {\n  bb1:\n    ret\n  }\n}\n";
        let err = parse(src).unwrap_err();
        assert!(err.msg.contains("out of order"), "{}", err.msg);
    }

    #[test]
    fn rejects_unknown_opcode() {
        let src = "module {\n  shared_mem_bytes 0\n  target_dtypes\n\n  func @f() -> void {\n  bb0:\n    frobnicate\n    ret\n  }\n}\n";
        let err = parse(src).unwrap_err();
        assert!(err.msg.contains("unknown opcode"), "{}", err.msg);
    }
}
