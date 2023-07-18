// use std::collections::HashMap;

use std::borrow::Cow;

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
pub enum Expr<'a> {
    Bool(bool),
    Num(f64),
    Ident(Cow<'a, str>),
    Neg(Box<Expr<'a>>),

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

#[derive(Debug)]
pub enum EvalError {
    NumberInLogic,
    BoolInMath,
    IdentNotFound(String),
}

impl<'a> Expr<'a>{
    fn into_owned(self) -> Expr<'static>{
        match self{
            Expr::Bool(b) => Expr::Bool(b),
            Expr::Num(n) => Expr::Num(n),
            Expr::Ident(cow) => Expr::Ident(Cow::Owned(cow.into_owned())),
            Expr::Neg(e) => Expr::Neg(Box::new(e.into_owned())),
            Expr::Mul(e1, e2) => Expr::Mul(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::Div(e1, e2) => Expr::Div(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::Add(e1, e2) => Expr::Add(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::Sub(e1, e2) => Expr::Sub(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::LessThan(e1, e2) => Expr::LessThan(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::GreaterThan(e1, e2) => Expr::GreaterThan(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::LessThanOrEqual(e1, e2) => Expr::LessThanOrEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::GreaterThanOrEqual(e1, e2) => Expr::GreaterThanOrEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::Equal(e1, e2) => Expr::Equal(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::NotEqual(e1, e2) => Expr::NotEqual(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::And(e1, e2) => Expr::And(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
            Expr::Or(e1, e2) => Expr::Or(Box::new(e1.into_owned()), Box::new(e2.into_owned())),
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
        let out = match self {
            Expr::Bool(b) => Expr::Bool(b),
            Expr::Num(n) => Expr::Num(n),
            Expr::Ident(ref name) => {
                let expr = lookup(name.clone()).ok_or_else(|| EvalError::IdentNotFound(name.to_string()))?;
                let expr = if expr != self {
                    expr.simplify_internal(lookup)?
                } else {
                    expr
                };
                expr
            }

            Expr::Neg(inner) => {
                let inner = inner.simplify_internal(lookup)?;
                match inner {
                    Expr::Num(n) => Expr::Num(-n),
                    Expr::Bool(_) => return Err(EvalError::BoolInMath),
                    _ => Expr::Neg(Box::new(inner)),
                }
            }

            Expr::Mul(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n), e) | (e, Expr::Num(n)) if n == 1.0 => e,
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Num(n1 * n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::Mul(Box::new(left), Box::new(right)),
                }
            }
            Expr::Div(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (e, Expr::Num(n)) if n == 1.0 => e,
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Num(n1 / n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::Div(Box::new(left), Box::new(right)),
                }
            }

            Expr::Add(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n), e) | (e, Expr::Num(n)) if n == 0.0 => e,
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Num(n1 + n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::Add(Box::new(left), Box::new(right)),
                }
            }
            Expr::Sub(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n), e) if n == 0.0 => Expr::Neg(Box::new(e)),
                    (e, Expr::Num(n)) if n == 0.0 => e,
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Num(n1 - n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::Sub(Box::new(left), Box::new(right)),
                }
            }

            Expr::LessThan(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 < n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::LessThan(Box::new(left), Box::new(right)),
                }
            }
            Expr::GreaterThan(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 > n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::GreaterThan(Box::new(left), Box::new(right)),
                }
            }
            Expr::LessThanOrEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 <= n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::LessThanOrEqual(Box::new(left), Box::new(right)),
                }
            }
            Expr::GreaterThanOrEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 >= n2),
                    (Expr::Bool(_), _) | (_, Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (left, right) => Expr::GreaterThanOrEqual(Box::new(left), Box::new(right)),
                }
            }
            Expr::Equal(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 == n2),
                    (Expr::Num(_), Expr::Bool(_)) => return Err(EvalError::BoolInMath),
                    (Expr::Bool(_), Expr::Num(_)) => return Err(EvalError::NumberInLogic),
                    (Expr::Bool(b1), Expr::Bool(b2)) => Expr::Bool(b1 == b2),
                    (left, right) => Expr::Equal(Box::new(left), Box::new(right)),
                }
            }
            Expr::NotEqual(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(n1), Expr::Num(n2)) => Expr::Bool(n1 != n2),
                    (Expr::Bool(b1), Expr::Bool(b2)) => Expr::Bool(b1 != b2),
                    (left, right) => Expr::NotEqual(Box::new(left), Box::new(right)),
                }
            }

            Expr::And(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(_), _) | (_, Expr::Num(_)) => return Err(EvalError::NumberInLogic),
                    (Expr::Bool(b1), Expr::Bool(b2)) => Expr::Bool(b1 && b2),
                    (left, right) => Expr::And(Box::new(left), Box::new(right)),
                }
            }

            Expr::Or(left, right) => {
                let left = left.simplify_internal(lookup)?;
                let right = right.simplify_internal(lookup)?;
                match (left, right) {
                    (Expr::Num(_), _) | (_, Expr::Num(_)) => return Err(EvalError::NumberInLogic),
                    (Expr::Bool(b1), Expr::Bool(b2)) => Expr::Bool(b1 || b2),
                    (left, right) => Expr::Or(Box::new(left), Box::new(right)),
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

fn parse_factor(input: &str) -> IResult<&str, Expr, NomError> {
    alt((parse_bool, parse_ident, parse_num, parse_neg, parse_parens))(input)
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

    let result = match op {
        "<=" => Expr::LessThanOrEqual(Box::new(initial), Box::new(expr)),
        ">=" => Expr::GreaterThanOrEqual(Box::new(initial), Box::new(expr)),
        "<" => Expr::LessThan(Box::new(initial), Box::new(expr)),
        ">" => Expr::GreaterThan(Box::new(initial), Box::new(expr)),
        "==" => Expr::Equal(Box::new(initial), Box::new(expr)),
        "!=" => Expr::NotEqual(Box::new(initial), Box::new(expr)),
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

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct If<'a>{
    #[serde(borrow)]
    condition: Expr<'a>,
    tokens: Vec<Token<'a>>,
    else_tokens: Option<Else<'a>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct For<'a>{
    ident: &'a str,
    range: Range<'a>,
    tokens: Vec<Token<'a>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub enum Token<'a> {
    Code(&'a str),
    Ident(&'a str),
    If(If<'a>),
    For(For<'a>)
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
        Ok((input, Some(Token::Code(code))))
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
            Token::Ident,
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

fn parse_inner(input: &str) -> IResult<&str, Vec<Token>, NomError>{
    let (input, inner) = 
        cut(delimited(
            preceded(multispace0, tag("{")),
            preceded(eat_newline, take_until_unbalanced('{', '}')),
            tag("}"),
        ))(input)?;

    let (inner, _) = eat_newline(trim_trailing_spaces(inner))?;

    let (_, inner_tokens) = parse_tokens(inner)?;

    Ok((input, inner_tokens))
}

fn parse_if(input: &str) -> IResult<&str, Token, NomError> {
    let (input, _) = tag("if")(input)?;

    let (input, condition) = cut(parse_expr)(input)?;
    
    // FIXME unwrap on simplify error. This should be propagated.
    // let condition = condition.simplify_without_ident()?;
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
            parse_if,
            parse_for,
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


#[derive(Debug)]
pub enum ExpansionError{
    IdentNotFound(String),
    SimplifyError(EvalError),
    NonBoolCondition(Expr<'static>),
    NonNumRange(Range<'static>),
}

impl<'a> From<EvalError> for ExpansionError{
    fn from(value: EvalError) -> Self {
        ExpansionError::SimplifyError(value)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Definition<'def> {
    Bool(bool),
    Int(i32),
    UInt(u32),
    Any(Cow<'def, str>),
    Float(f32),
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
        match value{
            Definition::Bool(def) => def.to_string(),
            Definition::Int(def) => def.to_string(),
            Definition::UInt(def) => {
                let mut out = def.to_string();
                out.push('u');
                out
            }
            Definition::Float(def) => format!("{:.1}", def),
            Definition::Any(def) => def.to_string(),
        }
    }
}


// FIXME This could be a regular function but I can't be arsed figuring out the traits and lifetimes
macro_rules! make_expr_lookup {
    ($func:ident) => {
    |s: Cow<'a, str>| -> Option<Expr<'a>> {
        let def = $func(s)?;
        match def{
            Definition::Bool(val) => Some(Expr::Bool(val)),
            Definition::Int(val) => Some(Expr::Num(val as f64)),
            Definition::UInt(val) => Some(Expr::Num(val as f64)),
            Definition::Float(val) => Some(Expr::Num(val as f64)),
            // FIXME
            Definition::Any(_val) => panic!("Maybe need to deal with this at some point"),
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
        .simplify_internal(&expr_lookup).unwrap();
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

fn process_internal<'a, 'def>(
    tokens: Vec<Token<'a>>,
    result: &mut String,
    lookup: &impl Fn(Cow<str>) -> Option<Definition<'def>>
) -> Result<(), ExpansionError>{
    for token in tokens{
        match token{
            Token::Code(code) => result.push_str(code),
            Token::Ident(name) => {
                let Some(shader_def) = lookup(name.into()) else { return Err(ExpansionError::IdentNotFound(name.to_string()))};
                let string = String::from(shader_def);
                if let Ok((_, tokens)) = parse_tokens(&string){
                    process_internal(tokens, result, lookup)?;
                } else{
                    result.push_str(&string);
                }
            },
            Token::If(if_tokens) => {
                process_if(if_tokens, result, lookup)?;
            },
            Token::For(for_tokens) => process_for(for_tokens, result, lookup)?,
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

