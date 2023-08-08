// use std::collections::HashMap;

use std::{borrow::Cow, fmt::Write};

// use crate::shaderpreprocessor::*;
use nom::{
    branch::alt,
    bytes::complete::{take_till, take_until, take_till1},
    character::complete::{alpha1, alphanumeric1, multispace0, char, space0, line_ending},
    combinator::{cut, map, recognize, opt},
    error::{ErrorKind, ParseError},
    multi::{many0, many0_count},
    number::complete::double,
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};

use nom_supreme::tag::complete::tag;

pub type NomError<'a> = nom_supreme::error::ErrorTree<&'a str>;


#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, Clone)]
pub enum Expr<'a>{
    Bool(bool),
    Num(f64),
    Ident(Cow<'a, str>),
    
    Neg(Box<Expr<'a>>),
    Not(Box<Expr<'a>>),

    Mul(Box<Expr<'a>>, Box<Expr<'a>>),
    Div(Box<Expr<'a>>, Box<Expr<'a>>),

    Add(Box<Expr<'a>>, Box<Expr<'a>>),
    Sub(Box<Expr<'a>>, Box<Expr<'a>>),

    LessThan(Box<Expr<'a>>, Box<Expr<'a>>),
    GreaterThan(Box<Expr<'a>>, Box<Expr<'a>>),
    LessThanOrEqual(Box<Expr<'a>>, Box<Expr<'a>>),
    GreaterThanOrEqual(Box<Expr<'a>>, Box<Expr<'a>>),
    Equal(Box<Expr<'a>>, Box<Expr<'a>>),
    NotEqual(Box<Expr<'a>>, Box<Expr<'a>>),

    And(Box<Expr<'a>>, Box<Expr<'a>>),

    Or(Box<Expr<'a>>, Box<Expr<'a>>),
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("A number was encountered in an expression where a boolean was expected")]
    NumberInLogic,
    #[error("A boolean was encountered in an expression where a number was expected")]
    BoolInMath,
    #[error("The identifier was not found {}", 0)]
    IdentNotFound(String),
}

impl<'a> Expr<'a>{
    fn into_owned(self) -> Expr<'static>{
        use Expr::*;
        match self{
            Bool(b) => Bool(b),
            Num(n) => Num(n),
            Ident(cow) => Ident(Cow::Owned(cow.into_owned())),
            Neg(e) => Neg(Box::new(e.into_owned())),
            Not(e) => Not(Box::new(e.into_owned())),
            Mul(e1, e2) => Mul(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Div(e1, e2) => Div(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Add(e1, e2) => Add(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Sub(e1, e2) => Sub(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            LessThan(e1, e2) => LessThan(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            GreaterThan(e1, e2) => GreaterThan(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            LessThanOrEqual(e1, e2) => LessThanOrEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            GreaterThanOrEqual(e1, e2) => GreaterThanOrEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Equal(e1, e2) => Equal(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            NotEqual(e1, e2) => NotEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            And(e1, e2) => And(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Or(e1, e2) => Or(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
        }
    }
}

impl<'a> Expr<'a> {
    pub fn simplify_without_ident(self) -> Result<Expr<'a>, EvalError> {
        self.simplify(|ident| Some(Expr::Ident(ident.into())))
    }

    pub fn simplify(self, lookup: impl Fn(Cow<'a, str>) -> Option<Expr>) -> Result<Expr<'a>, EvalError> {
        self.simplify_internal(&lookup)
    }

    fn simplify_internal(
        self,
        lookup: &impl Fn(Cow<'a, str>) -> Option<Expr>,
    ) -> Result<Expr<'a>, EvalError> {
        use Expr::*;
        let out = match self {
            Bool(b) => Bool(b),
            Num(n) => Num(n),
            Ident(ref name) => {
                let expr = lookup(name.clone()).ok_or_else(|| EvalError::IdentNotFound(name.to_string()))?;
                let expr = if expr != self {
                    expr.simplify_internal(lookup)?
                } else {
                    expr
                };
                expr
            }

            Neg(inner) => {
                let inner = inner.simplify_internal(lookup)?;
                match inner {
                    Num(n) => Num(-n),
                    Bool(_) => return Err(EvalError::BoolInMath),
                    _ => Neg(Box::new(inner)),
                }
            }
            Not(inner) => {
                let inner = inner.simplify_internal(lookup)?;
                match inner {
                    Num(_) => return Err(EvalError::NumberInLogic),
                    Bool(b) => Bool(!b),
                    _ => Not(Box::new(inner)),
                }
            }

            Mul(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n), e) | (e, Num(n)) if n == 1.0 => e,
                    (Num(n1), Num(n2)) => Num(n1 * n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Mul(Box::new(left), Box::new(right)),
                }
            }
            Div(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (e, Num(n)) if n == 1.0 => e,
                    (Num(n1), Num(n2)) => Num(n1 / n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Div(Box::new(left), Box::new(right)),
                }
            }

            Add(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n), e) | (e, Num(n)) if n == 0.0 => e,
                    (Num(n1), Num(n2)) => Num(n1 + n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Add(Box::new(left), Box::new(right)),
                }
            }
            Sub(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n), e) if n == 0.0 => Neg(Box::new(e)),
                    (e, Num(n)) if n == 0.0 => e,
                    (Num(n1), Num(n2)) => Num(n1 - n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Sub(Box::new(left), Box::new(right)),
                }
            }

            LessThan(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 < n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => LessThan(Box::new(left), Box::new(right)),
                }
            }
            GreaterThan(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 > n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => GreaterThan(Box::new(left), Box::new(right)),
                }
            }
            LessThanOrEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 <= n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => LessThanOrEqual(Box::new(left), Box::new(right)),
                }
            }
            GreaterThanOrEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 >= n2),
                    (Bool(_), _) | (_, Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => GreaterThanOrEqual(Box::new(left), Box::new(right)),
                }
            }
            Equal(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 == n2),
                    (Bool(b1), Bool(b2)) => Bool(b1 == b2),
                    (Num(_), Bool(_)) => return Err(EvalError::BoolInMath),
                    (Bool(_), Num(_)) => return Err(EvalError::NumberInLogic),
                    (left, right) => Equal(Box::new(left), Box::new(right)),
                }
            }
            NotEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Num(n1), Num(n2)) => Bool(n1 != n2),
                    (Bool(b1), Bool(b2)) => Bool(b1 != b2),
                    (left, right) => NotEqual(Box::new(left), Box::new(right)),
                }
            }

            And(left, right) => {
                let left = left.simplify_internal(lookup)?;
                if matches!(left, Bool(false)){
                    Bool(false)
                } else {
                    let right = right.simplify_internal(lookup)?;
                    match (left, right) {
                        (Num(_), _) | (_, Num(_)) => return Err(EvalError::NumberInLogic),
                        (Bool(b1), Bool(b2)) => Bool(b1 && b2),
                        (left, right) => And(Box::new(left), Box::new(right)),
                    }
                }
            }

            Or(left, right) => {
                let left = left.simplify_internal(lookup)?;
                if matches!(left, Bool(true)){
                    Bool(true)
                } else {
                    let right = right.simplify_internal(lookup)?;
                    match (left, right) {
                        (Num(_), _) | (_, Num(_)) => return Err(EvalError::NumberInLogic),
                        (Bool(b1), Bool(b2)) => Bool(b1 || b2),
                        (left, right) => Or(Box::new(left), Box::new(right)),
                    }
                }
            }
        };
        return Ok(out);
    }
}

fn parse_bool(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, result) = preceded(
        multispace0,
        alt((
            map(tag("true"), |_| Expr::Bool(true)),
            map(tag("false"), |_| Expr::Bool(false)),
        )),
    )(input)?;

    Ok((input, result))
}

fn parse_num(input: &str) -> IResult<&str, Expr, NomError> {
    preceded(multispace0, map(double, Expr::Num))(input)
}

fn parse_ident(input: &str) -> IResult<&str, Expr, NomError> {
    preceded(
        multispace0,
        map(
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0_count(alt((alphanumeric1, tag("_")))),
            )),
            |s: &str| Expr::Ident(s.into()),
        ),
    )(input)
}

fn parse_parens(input: &str) -> IResult<&str, Expr, NomError> {
    preceded(
        multispace0,
        delimited(
            preceded(multispace0, char('(')),
            parse_expr,
            preceded(multispace0, char(')')),
        ),
    )(input)
}

fn parse_neg(input: &str) -> IResult<&str, Expr, NomError> {
    map(
        preceded(multispace0, pair(char('-'), parse_factor)),
        |(_, expr)| Expr::Neg(Box::new(expr)),
    )(input)
}

fn parse_not(input: &str) -> IResult<&str, Expr, NomError> {
    map(
        preceded(multispace0, pair(char('!'), parse_factor)),
        |(_, expr)| Expr::Not(Box::new(expr)),
    )(input)
}

fn parse_factor(input: &str) -> IResult<&str, Expr, NomError> {
    alt((parse_bool, parse_ident, parse_num, parse_neg, parse_not, parse_parens))(input)
}

fn parse_mul_div(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, init) = parse_factor(input)?;
    let (input, ops) = nom::multi::many0(pair(
        preceded(multispace0, alt((char('*'), char('/')))),
        preceded(multispace0, parse_factor),
    ))(input)?;

    let expr = ops.into_iter().fold(init, |acc, (op, factor)| {
        if op == '*' {
            Expr::Mul(Box::new(acc), Box::new(factor))
        } else {
            Expr::Div(Box::new(acc), Box::new(factor))
        }
    });

    Ok((input, expr))
}

fn parse_add_sub(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, init) = parse_mul_div(input)?;
    let (input, ops) = nom::multi::many0(pair(
        preceded(multispace0, alt((char('+'), char('-')))),
        preceded(multispace0, parse_mul_div),
    ))(input)?;

    let expr = ops.into_iter().fold(init, |acc, (op, term)| {
        if op == '+' {
            Expr::Add(Box::new(acc), Box::new(term))
        } else {
            Expr::Sub(Box::new(acc), Box::new(term))
        }
    });

    Ok((input, expr))
}

fn parse_comparison(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, initial) = parse_add_sub(input)?;
    
    let (input, mut comparisons) = many0(pair(
        preceded(
            multispace0,
            alt((
                tag("<="),
                tag(">="),
                tag("<"),
                tag(">"),
                tag("=="),
                tag("!="),
            )),
        ),
        preceded(multispace0, parse_add_sub),
    ))(input)?;

    let (op, expr) = match comparisons.len() {
        0 => return Ok((input, initial)),
        1 => comparisons.remove(0),
        _ => {
            // return Err(nom::Err::Failure(nom::error::Error::new(
            //     input,
            //     nom::error::ErrorKind::TooLarge,
            // )))
            return Err(nom::Err::Failure(NomError::from_error_kind(input, ErrorKind::TooLarge)))
        }
    };

    use Expr::*;
    let result = match op {
        "<=" => LessThanOrEqual(Box::new(initial), Box::new(expr)),
        ">=" => GreaterThanOrEqual(Box::new(initial), Box::new(expr)),
        "<" => LessThan(Box::new(initial), Box::new(expr)),
        ">" => GreaterThan(Box::new(initial), Box::new(expr)),
        "==" => Equal(Box::new(initial), Box::new(expr)),
        "!=" => NotEqual(Box::new(initial), Box::new(expr)),
        _ => unreachable!(),
    };

    Ok((input, result))
}

fn parse_and(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, init) = parse_comparison(input)?;
    let (input, ops) = nom::multi::many0(pair(
        preceded(multispace0, tag("&&")),
        preceded(multispace0, parse_comparison),
    ))(input)?;

    let expr = ops
        .into_iter()
        .fold(init, |acc, term| Expr::And(Box::new(acc), Box::new(term.1)));

    Ok((input, expr))
}

fn parse_or(input: &str) -> IResult<&str, Expr, NomError> {
    let (input, init) = parse_and(input)?;
    let (input, ops) = nom::multi::many0(pair(
        preceded(multispace0, tag("||")),
        preceded(multispace0, parse_and),
    ))(input)?;

    let expr = ops
        .into_iter()
        .fold(init, |acc, term| Expr::Or(Box::new(acc), Box::new(term.1)));

    Ok((input, expr))
}

pub fn parse_expr(input: &str) -> IResult<&str, Expr, NomError> {
    parse_or(input)
}

fn parse_token_expr(input: &str) -> IResult<&str, Token, NomError>{
    let (input, _) = preceded(multispace0, tag("expr"))(input)?;

    let (input, inner) = cut(get_inner)(input)?;

    let (_, expr) = cut(parse_expr)(inner)?;

    let expr = Token::Expr(expr);

    Ok((input, expr))
}

// https://stackoverflow.com/questions/70630556/parse-allowing-nested-parentheses-in-nom
pub fn take_until_unbalanced(
    opening_bracket: char,
    closing_bracket: char,
) -> impl Fn(&str) -> IResult<&str, &str, NomError> {
    move |i: &str| {
        let mut index = 0;
        let mut bracket_counter = 0;
        while let Some(n) = &i[index..].find(&[opening_bracket, closing_bracket, '\\'][..]) {
            index += n;
            let mut it = i[index..].chars();
            match it.next().unwrap_or_default() {
                c if c == '\\' => {
                    // Skip the escape char `\`.
                    index += '\\'.len_utf8();
                    // Skip also the following char.
                    let c = it.next().unwrap_or_default();
                    index += c.len_utf8();
                }
                c if c == opening_bracket => {
                    bracket_counter += 1;
                    index += opening_bracket.len_utf8();
                }
                c if c == closing_bracket => {
                    // Closing bracket.
                    bracket_counter -= 1;
                    index += closing_bracket.len_utf8();
                }
                // Can not happen.
                _ => unreachable!(),
            };
            // We found the unmatched closing bracket.
            if bracket_counter == -1 {
                // We do not consume it.
                index -= closing_bracket.len_utf8();
                return Ok((&i[index..], &i[0..index]));
            };
        }

        if bracket_counter == 0 {
            Ok(("", i))
        } else {
            Err(nom::Err::Error(NomError::from_error_kind(
                i,
                ErrorKind::TakeUntil,
            )))
            // Err(nom::Err::Error(Error::from_error_kind(
            //     i,
            //     ErrorKind::TakeUntil,
            // )))
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub enum Range<'a> {
    #[serde(borrow)]
    Exclusive((Expr<'a>, Expr<'a>)),
    Inclusive((Expr<'a>, Expr<'a>)),
}

impl<'a> Range<'a>{
    fn into_owned(self) -> Range<'static>{
        match self{
            Range::Exclusive((expr1, expr2)) => Range::Exclusive((expr1.into_owned(), expr2.into_owned())),
            Range::Inclusive((expr1, expr2)) => Range::Inclusive((expr1.into_owned(), expr2.into_owned())),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub enum Else<'a>{
    #[serde(borrow)]
    Block(Vec<Token<'a>>),
    If(Box<If<'a>>),
}

impl<'a> Else<'a>{
    fn into_owned(self) -> Else<'static>{
        match self{
            Else::Block(tokens) => Else::Block(vec_to_owned(tokens)),
            Else::If(if_tok) => Else::If(Box::new((*if_tok).into_owned())),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct If<'a>{
    #[serde(borrow)]
    condition: Expr<'a>,
    tokens: Vec<Token<'a>>,
    else_tokens: Option<Else<'a>>,
}

impl<'a> If<'a>{
    fn into_owned(self) -> If<'static>{
        If{
            condition: self.condition.into_owned(),
            tokens: vec_to_owned(self.tokens),
            else_tokens: self.else_tokens.map(|e| e.into_owned())
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct For<'a>{
    #[serde(borrow)]
    ident: Cow<'a, str>,
    range: Range<'a>,
    tokens: Vec<Token<'a>>,
}

impl<'a> For<'a>{
    fn into_owned(self) -> For<'static>{
        For{
            ident: Cow::Owned(self.ident.into_owned()),
            range: self.range.into_owned(),
            tokens: vec_to_owned(self.tokens),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct NestedFor<'a>{
    #[serde(borrow)]
    running_ident: Cow<'a, str>,
    total_ident: Cow<'a, str>,
    to_nest: Vec<Token<'a>>,
    pre: Vec<Token<'a>>,
    inner: Vec<Token<'a>>,
}

impl<'a> NestedFor<'a>{
    fn into_owned(self) -> NestedFor<'static>{
        NestedFor{
            running_ident: Cow::Owned(self.running_ident.into_owned()),
            total_ident: Cow::Owned(self.total_ident.into_owned()),
            to_nest: vec_to_owned(self.to_nest),
            pre: vec_to_owned(self.pre),
            inner: vec_to_owned(self.inner),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Concat<'a>{
    #[serde(borrow)]
    ident: Cow<'a, str>,
    range: Range<'a>,
    tokens: Vec<Token<'a>>,
    separator: Vec<Token<'a>>,
}

impl<'a> Concat<'a>{
    fn into_owned(self) -> Concat<'static>{
        Concat{
            ident: Cow::Owned(self.ident.into_owned()),
            range: self.range.into_owned(),
            tokens: vec_to_owned(self.tokens),
            separator: vec_to_owned(self.separator),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub enum Token<'a> {
    #[serde(borrow)]
    Code(Cow<'a, str>),
    Ident(Cow<'a, str>),
    Expr(Expr<'a>),
    If(If<'a>),
    For(For<'a>),
    NestedFor(NestedFor<'a>),
    Concat(Concat<'a>)
}

pub fn vec_to_owned<'a>(tokens: Vec<Token<'a>>) -> Vec<Token<'static>>{
    tokens.into_iter().map(|token| token.into_owned()).collect()
}

impl<'a> Token<'a>{
    fn into_owned(self) -> Token<'static>{
        use Token::*;
        match self{
            Code(cow) => Code(Cow::Owned(cow.to_string())),
            Ident(cow) => Ident(Cow::Owned(cow.to_string())),
            Expr(expr) => Expr(expr.into_owned()),
            If(if_tok) => If(if_tok.into_owned()),
            For(for_tok) => For(for_tok.into_owned()),
            NestedFor(nested) => NestedFor(nested.into_owned()),
            Concat(concat) => Concat(concat.into_owned()),
        }
    }
}

fn parse_comment(input: &str) -> IResult<&str, &str, NomError> {
    recognize(
        tuple((
            tag("//"),
            take_till(|c| c == '\n'),
            opt(char('\n')),
        ))
    )(input)
}

fn parse_shader_code(input: &str) -> IResult<&str, Option<Token>, NomError> {
    let (input, code) = recognize(many0(
        alt((
            take_till1(|c| c == '#' || c == '/'),
            alt((parse_comment, tag("/"))),
        )),
    ))(input)?;
    
    if code.is_empty() {
        Ok((input, None))
    } else {
        let code = trim_trailing_spaces(code);
        Ok((input, Some(Token::Code(code.into()))))
    }
}

fn parse_ident_token(input: &str) -> IResult<&str, Token, NomError> {
    preceded(
        multispace0,
        map(
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0_count(alt((alphanumeric1, tag("_")))),
            )),
            |s: &str| Token::Ident(s.into()),
        ),
    )(input)
}

fn eat_newline(input: &str) -> IResult<&str, (), NomError>{
    let (input, _) = opt(preceded(space0, line_ending))(input)?;
    Ok((input, ()))
}

pub fn trim_trailing_spaces(input: &str) -> &str {
    let mut chars = input.chars().rev();
    let mut trailing_spaces = 0;

    while let Some(ch) = chars.next() {
        match ch {
            ' ' | '\t' => trailing_spaces += 1,
            '\n' => {
                break
            },
            _ => {
                trailing_spaces = 0;
                break
            }
        }
    }
    &input[..input.len() - trailing_spaces]
}

fn get_inner(input: &str) -> IResult<&str, &str, NomError>{
    let (input, inner) = 
        delimited(
            preceded(multispace0, tag("{")),
            preceded(eat_newline, take_until_unbalanced('{', '}')),
            tag("}"),
        )(input)?;

    let (inner, _) = eat_newline(trim_trailing_spaces(inner))?;
    
    Ok((input, inner))
}

fn parse_inner(input: &str) -> IResult<&str, Vec<Token>, NomError>{
    
    let (input, inner) = cut(get_inner)(input)?;

    let (_, inner_tokens) = cut(parse_tokens)(inner)?;

    Ok((input, inner_tokens))
}

fn parse_if(input: &str) -> IResult<&str, Token, NomError> {
    let (input, _) = tag("if")(input)?;

    let (input, condition) = cut(parse_expr)(input)?;
    
    // FIXME unwrap on simplify error. This should be propagated.
    let condition = condition.simplify_without_ident().unwrap();

    let (input, inner_tokens) = cut(parse_inner)(input)?;

    let (input, else_tag) = opt(
        preceded(
            multispace0,
            tag("#else"),
        )
    )(input)?;

    let (input, else_tokens) = match else_tag{
        Some(_) => {
            cut(alt((
                map(preceded(multispace0, parse_if), |res| {
                    let Token::If(res) = res else { unreachable!() };
                    Some(Else::If(Box::new(res)))
                }),
                map(parse_inner, |res| Some(Else::Block(res))),
            )))(input)?
        },
        None => (input, None),
    };

    Ok((
        input,
        Token::If(If{
            condition,
            tokens: inner_tokens,
            else_tokens,
        }),
    ))
}

fn parse_range(input: &str) -> IResult<&str, Range, NomError> {
    let (input, first_expr_str) = cut(take_until(".."))(input)?;
    
    let (_, exp1) = parse_expr(first_expr_str)?;
    let (input, ty) = cut(alt((tag("..="), tag(".."))))(input)?;
    let (input, exp2) = parse_expr(input)?;
    Ok((
        input,
        match ty {
            "..=" => Range::Inclusive((exp1, exp2)),
            ".." => Range::Exclusive((exp1, exp2)),
            _ => unreachable!(),
        },
    ))
}

fn parse_for(input: &str) -> IResult<&str, Token, NomError> {
    let (input, _) = tag("for")(input)?;

    let (input, Token::Ident(ident)) = cut(parse_ident_token)(input)? else { unreachable!() };

    let (input, _) = cut(
        preceded(multispace0,
        tag("in"))
    )(input)?;

    let (input, range) = cut(parse_range)(input)?;

    let (input, inner) = cut(parse_inner)(input)?;

    let result = Token::For(For{
        ident,
        range,
        tokens: inner,
    });
    Ok((input, result))
}

fn parse_concat(input: &str) -> IResult<&str, Token, NomError> {
    let (input, _) = tag("concat")(input)?;

    let (input, Token::Ident(ident)) = cut(parse_ident_token)(input)? else { unreachable!() };

    let (input, _) = cut(
        preceded(multispace0,
        tag("in"))
    )(input)?;

    let (input, range) = cut(parse_range)(input)?;

    let (input, inner) = cut(parse_inner)(input)?;
    
    let (input, separator) = cut(parse_inner)(input)?;

    let result = Token::Concat(Concat{
        ident,
        range,
        tokens: inner,
        separator,
    });
    Ok((input, result))
}

fn parse_nest(input: &str) -> IResult<&str, Token, NomError> {
    let (input, _) = tag("nest")(input)?;

    let (input, Token::Ident(running_ident)) = cut(parse_ident_token)(input)? else { unreachable!() };

    let (input, _) = cut(preceded(multispace0, char('=')))(input)?;

    let (input, Token::Ident(total_ident)) = cut(parse_ident_token)(input)? else { unreachable!() };
    
    let (input, to_nest) = cut(parse_inner)(input)?;

    let (input, _) = cut(preceded(multispace0, tag("#pre")))(input)?;
    
    let (input, pre) = cut(parse_inner)(input)?;
    
    let (input, _) = cut(preceded(multispace0, tag("#inner")))(input)?;
    
    let (input, inner) = cut(parse_inner)(input)?;

    let result = Token::NestedFor(NestedFor{
        running_ident,
        total_ident,
        to_nest,
        pre,
        inner,
    });
    Ok((input, result))
}

pub fn parse_tokens(mut input: &str) -> IResult<&str, Vec<Token>, NomError> {
    let mut out = Vec::new();

    // Consume initial shader code, up to the first "#"
    let (new_input, code) = parse_shader_code(input)?;
    if let Some(code) = code{
        out.push(code);
    }
    input = new_input;

    while !input.is_empty() {
        let (new_input, _) = char('#')(input)?;
        input = new_input;
        // Parse directive
        let (new_input, token) = alt((
            parse_token_expr,
            parse_if,
            parse_for,
            parse_concat,
            parse_nest,
            parse_ident_token,
        ))(input)?;
        out.push(token);
        let (new_input, code) = parse_shader_code(new_input)?;
        if let Some(code) = code{
            
            out.push(code);
        }
        input = new_input;
    }
    Ok((input, out))
}


#[derive(Debug, thiserror::Error)]
pub enum ExpansionError{
    #[error("The identifier was not found")]
    IdentNotFound(String),
    #[error("There was a problem with evaluating an expression")]
    SimplifyError(#[from] EvalError),
    #[error("A condition simplified to something that wasn't a boolean")]
    NonBoolCondition(Expr<'static>),
    #[error("A range contained something that wasn't a number")]
    NonNumRange(Range<'static>),
    #[error("An explicit expression contained something that wasn't a boolean or a number")]
    NonBoolOrNumExpr(Expr<'static>),
    #[error("A number was expected for this definition")]
    ExpectedNumber(Definition<'static>)
}

// impl<'a> From<EvalError> for ExpansionError{
//     fn from(value: EvalError) -> Self {
//         ExpansionError::SimplifyError(value)
//     }
// }

#[derive(Clone, PartialEq, Debug)]
pub enum Definition<'def> {
    Bool(bool),
    Int(i32),
    UInt(u32),
    Any(Cow<'def, str>),
    Float(f32),
}

impl<'def> Definition<'def>{
    fn new_owned(&self) -> Definition<'static>{
        use Definition::*;
        match self{
            Bool(v) => Bool(*v),
            Int(v) => Int(*v),
            UInt(v) => UInt(*v),
            Any(cow) => Any(Cow::Owned(cow.to_string())),
            Float(v) => Float(*v),
        }
    }
}

impl<'a> From<bool> for Definition<'a>{
    fn from(value: bool) -> Self {
        Definition::Bool(value)
    }
}

impl<'a> From<i32> for Definition<'a>{
    fn from(value: i32) -> Self {
        Definition::Int(value)
    }
}

impl<'a> From<u32> for Definition<'a>{
    fn from(value: u32) -> Self {
        Definition::UInt(value)
    }
}

impl<'a> From<&'a str> for Definition<'a>{
    fn from(value: &'a str) -> Self {
        Definition::Any(value.into())
    }
}

impl<'a> From<String> for Definition<'a>{
    fn from(value: String) -> Self {
        Definition::Any(value.into())
    }
}

impl<'a> From<f32> for Definition<'a>{
    fn from(value: f32) -> Self {
        Definition::Float(value)
    }
}

impl<'def> Default for Definition<'def>{
    fn default() -> Self {
        Self::Any("".into())
    }
}

impl<'def> From<Definition<'def>> for String {
    fn from(value: Definition) -> Self {
        let mut out = String::new();
        value.insert_in_string(&mut out).unwrap();
        out
    }
}

impl<'def> Definition<'def>{
    fn insert_in_string(&self, target: &mut String) -> Result<(), std::fmt::Error>{
        match self{
            Definition::Bool(def) => write!(target, "{def}"),
            Definition::Int(def) => write!(target, "{def}"),
            Definition::UInt(def) => write!(target, "{def}u"),
            Definition::Float(def) => write!(target, "{def:.1}"),
            Definition::Any(def) => write!(target, "{def}"),
        }
    }
}


// FIXME This could be a regular function but I can't be arsed figuring out the traits and lifetimes
macro_rules! make_expr_lookup {
    ($func:ident) => {
    |s: Cow<'a, str>| -> Option<Expr<'a>> {
        let def = $func(s)?;
        use Definition::*;
        match def{
            Bool(val) => Some(Expr::Bool(val)),
            Int(val) => Some(Expr::Num(val as f64)),
            UInt(val) => Some(Expr::Num(val as f64)),
            Float(val) => Some(Expr::Num(val as f64)),
            // FIXME
            Any(_val) => panic!("Maybe need to deal with this at some point"),
        }
    }
    };
}

fn process_if<'a, 'def>(
    input: If<'a>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>
) -> Result<(), ExpansionError>{
    let expr_lookup = make_expr_lookup!(lookup);

    let If{
        condition,
        tokens,
        else_tokens,
    } = input;

    let condition = condition
        .simplify_internal(&expr_lookup)?;
    match condition{
        Expr::Bool(true) => process_internal(tokens, result, lookup),
        Expr::Bool(false) => {
            match else_tokens{
                Some(Else::Block(tokens)) => {
                    process_internal(tokens, result, lookup)
                },
                Some(Else::If(new_if)) => {
                    process_if(*new_if, result, lookup)
                },
                None => Ok(()),
            }
        },
        _ => return Err(ExpansionError::NonBoolCondition(condition.into_owned())),
    }
}

fn process_for<'a, 'def>(
    input: For<'a>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>,
) -> Result<(), ExpansionError>{
    let For{
        ident,
        range,
        tokens,
    } = input;

    let expr_lookup = make_expr_lookup!(lookup);

    let range = match range{
        Range::Exclusive((expr1, expr2)) => {
            let expr1 = expr1.simplify(expr_lookup)?;
            let expr2 = expr2.simplify(expr_lookup)?;
            Range::Exclusive((expr1, expr2))
        },
        Range::Inclusive((expr1, expr2)) => {
            let expr1 = expr1.simplify(expr_lookup)?;
            let expr2 = expr2.simplify(expr_lookup)?;
            Range::Inclusive((expr1, expr2))
        },
    };

    let iter = match range{
        Range::Exclusive((Expr::Num(start), Expr::Num(end))) => {
            Box::new(start as isize..end as isize) as Box<dyn Iterator<Item = isize>>
        },
        Range::Inclusive((Expr::Num(start), Expr::Num(end))) => {
            Box::new(start as isize..=end as isize) as Box<dyn Iterator<Item = isize>>
        },
        _ => return Err(ExpansionError::NonNumRange(range.into_owned()))
    };

    for val in iter{
        let new_lookup = Box::new(|s: Cow<str>| -> Option<Definition<'def>>{
            if s == ident{
                Some(Definition::Int(val as i32))
            } else {
                lookup(s)
            }
        }) as Box<dyn Fn(Cow<str>) -> Option<Definition<'def>>>;
        process_internal(tokens.clone(), result, &new_lookup)?;
    }
    Ok(())
}

fn process_nested_for<'a, 'def>(
    input: NestedFor<'a>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>,
) -> Result<(), ExpansionError>{
    let NestedFor { running_ident, total_ident, to_nest, pre, inner } = input;
    
    let total_depth = lookup(total_ident.clone()).ok_or_else(|| ExpansionError::IdentNotFound(total_ident.to_string()))?;
    
    let total_depth = match total_depth{
        Definition::Bool(_) | Definition::Any(_) => return Err(ExpansionError::ExpectedNumber(total_depth.new_owned())),
        Definition::Int(num) => num as usize,
        Definition::UInt(num) => num as usize,
        Definition::Float(num) => num as usize,
    };
    
    for val in 0..total_depth{
        let new_lookup = Box::new(|s: Cow<str>| -> Option<Definition<'def>>{
            if s == running_ident{
                Some(Definition::Int(val as i32))
            } else {
                lookup(s)
            }
        }) as Box<dyn Fn(Cow<str>) -> Option<Definition<'def>>>;
        
        process_internal(to_nest.clone(), result, &new_lookup)?;
        result.push_str("{\n");
        process_internal(pre.clone(), result, &new_lookup)?;
        result.push('\n');
    }
    
    process_internal(inner, result, lookup)?;
    
    for _ in 0..total_depth{
        result.push_str("\n}")
    }
    
    Ok(())
}

fn process_concat<'a, 'def>(
    input: Concat<'a>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>,
) -> Result<(), ExpansionError>{
    let Concat { ident, range, tokens, separator } = input;
    
    let expr_lookup = make_expr_lookup!(lookup);

    let range = match range{
        Range::Exclusive((expr1, expr2)) => {
            let expr1 = expr1.simplify(expr_lookup)?;
            let expr2 = expr2.simplify(expr_lookup)?;
            Range::Exclusive((expr1, expr2))
        },
        Range::Inclusive((expr1, expr2)) => {
            let expr1 = expr1.simplify(expr_lookup)?;
            let expr2 = expr2.simplify(expr_lookup)?;
            Range::Inclusive((expr1, expr2))
        },
    };

    let iter = match range{
        Range::Exclusive((Expr::Num(start), Expr::Num(end))) => {
            Box::new(start as isize..end as isize) as Box<dyn Iterator<Item = isize>>
        },
        Range::Inclusive((Expr::Num(start), Expr::Num(end))) => {
            Box::new(start as isize..=end as isize) as Box<dyn Iterator<Item = isize>>
        },
        _ => return Err(ExpansionError::NonNumRange(range.into_owned()))
    };
    let mut iter = iter.peekable();

    while let Some(val) = iter.next(){
        let new_lookup = Box::new(|s: Cow<str>| -> Option<Definition<'def>>{
            if s == ident{
                Some(Definition::Int(val as i32))
            } else {
                lookup(s)
            }
        }) as Box<dyn Fn(Cow<str>) -> Option<Definition<'def>>>;
        
        process_internal(tokens.clone(), result, &new_lookup)?;

        if iter.peek().is_some(){
            process_internal(separator.clone(), result, &new_lookup)?;
        }
        
    }
    Ok(())
}


fn process_internal<'a, 'def>(
    tokens: Vec<Token<'a>>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>
) -> Result<(), ExpansionError>{
    for token in tokens{
        match token{
            Token::Code(code) => result.push_str(&code),
            Token::Ident(name) => {
                let Some(shader_def) = lookup(name.clone()) else { return Err(ExpansionError::IdentNotFound(name.to_string()))};
                let string = String::from(shader_def);
                if let Ok((_, tokens)) = parse_tokens(&string){
                    process_internal(tokens, result, lookup)?;
                } else{
                    write!(result, "{}", string).expect("We couldn't write to a string?");
                }
            },
            Token::Expr(expr) => {
                let expr_lookup = make_expr_lookup!(lookup);
                let simplified_expr = expr.simplify_internal(&expr_lookup)?;
                match simplified_expr{
                    Expr::Bool(b) => write!(result, "{}", b).expect("We couldn't write to a string?"),
                    Expr::Num(n) => write!(result, "{}", n).expect("We couldn't write to a string?"),
                    _  => return Err(ExpansionError::NonBoolOrNumExpr(simplified_expr.into_owned())),
                }
            },
            Token::If(if_tokens) => {
                process_if(if_tokens, result, lookup)?;
            },
            Token::For(for_tokens) => process_for(for_tokens, result, lookup)?,
            Token::NestedFor(nested_for) => process_nested_for(nested_for, result, lookup)?,
            Token::Concat(concat) => process_concat(concat, result, lookup)?,
        }
    }
    Ok(())
}

pub fn process<'def>(tokens: Vec<Token>, lookup: impl Fn(Cow<str>) -> Option<Definition<'def>>) -> Result<String, ExpansionError>{
    let mut result = String::new();
    
    process_internal(
        tokens,
        &mut result,
        &lookup,
    )?;

    Ok(result)
}

