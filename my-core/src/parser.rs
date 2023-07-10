// use std::collections::HashMap;

// use crate::shaderpreprocessor::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till},
    character::complete::char,
    character::complete::{alpha1, alphanumeric1, multispace0},
    combinator::{cut, map, recognize},
    error::{context, Error, ErrorKind, ParseError},
    multi::{many0, many0_count},
    number::complete::double,
    sequence::{delimited, pair, preceded, terminated},
    IResult,
};

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expr<'a> {
    Bool(bool),
    Num(f64),
    Ident(&'a str),
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
    IdentNotFound,
}

impl<'a> Expr<'a> {
    pub fn simplify_without_ident(self) -> Result<Expr<'a>, EvalError> {
        self.simplify(|ident| Some(Expr::Ident(ident)))
    }

    pub fn simplify(self, lookup: impl Fn(&str) -> Option<Expr>) -> Result<Expr<'a>, EvalError> {
        self.simplify_internal(&lookup)
    }

    fn simplify_internal(
        self,
        lookup: &impl Fn(&str) -> Option<Expr>,
    ) -> Result<Expr<'a>, EvalError> {
        let out = match self {
            Expr::Bool(b) => Expr::Bool(b),
            Expr::Num(n) => Expr::Num(n),
            Expr::Ident(name) => {
                let expr = lookup(name).ok_or(EvalError::IdentNotFound)?;
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

fn parse_bool(input: &str) -> IResult<&str, Expr> {
    let (input, result) = preceded(
        multispace0,
        alt((
            map(tag("true"), |_| Expr::Bool(true)),
            map(tag("false"), |_| Expr::Bool(false)),
        )),
    )(input)?;

    Ok((input, result))
}

fn parse_num(input: &str) -> IResult<&str, Expr> {
    preceded(multispace0, map(double, Expr::Num))(input)
}

fn parse_ident(input: &str) -> IResult<&str, Expr> {
    preceded(
        multispace0,
        map(
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0_count(alt((alphanumeric1, tag("_")))),
            )),
            Expr::Ident,
        ),
    )(input)
}

fn parse_parens(input: &str) -> IResult<&str, Expr> {
    preceded(
        multispace0,
        delimited(
            preceded(multispace0, char('(')),
            parse_expr,
            preceded(multispace0, char(')')),
        ),
    )(input)
}

fn parse_neg(input: &str) -> IResult<&str, Expr> {
    map(
        preceded(multispace0, pair(char('-'), parse_factor)),
        |(_, expr)| Expr::Neg(Box::new(expr)),
    )(input)
}

fn parse_factor(input: &str) -> IResult<&str, Expr> {
    alt((parse_num, parse_neg, parse_bool, parse_ident, parse_parens))(input)
}

fn parse_mul_div(input: &str) -> IResult<&str, Expr> {
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

fn parse_add_sub(input: &str) -> IResult<&str, Expr> {
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

fn parse_comparison(input: &str) -> IResult<&str, Expr> {
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
            return Err(nom::Err::Failure(nom::error::Error::new(
                input,
                nom::error::ErrorKind::TooLarge,
            )))
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

fn parse_and(input: &str) -> IResult<&str, Expr> {
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

fn parse_or(input: &str) -> IResult<&str, Expr> {
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

pub fn parse_expr(input: &str) -> IResult<&str, Expr> {
    parse_or(input)
}

pub fn take_until_unbalanced(
    opening_bracket: char,
    closing_bracket: char,
) -> impl Fn(&str) -> IResult<&str, &str> {
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
            Err(nom::Err::Error(Error::from_error_kind(
                i,
                ErrorKind::TakeUntil,
            )))
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Range<'a> {
    #[serde(borrow)]
    Exclusive((Expr<'a>, Expr<'a>)),
    Inclusive((Expr<'a>, Expr<'a>)),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Token<'a> {
    Code(&'a str),
    Ident(&'a str),
    If {
        condition: Expr<'a>,
        tokens: Vec<Token<'a>>,
    },
    // For {
    //     tokens: Vec<Token<'a>>,
    //     ident: &'a str,
    //     range: &'a str,
    // },
}

fn parse_shader_code(input: &str) -> IResult<&str, Token> {
    let (input, code) = take_till(|c| c == '#')(input)?;
    Ok((input, Token::Code(code)))
}

fn parse_ident_token(input: &str) -> IResult<&str, Token> {
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

fn parse_if(input: &str) -> IResult<&str, Token> {
    let (input, _) = tag("if")(input)?;

    let (input, condition) = cut(parse_expr)(input)?;
    let condition = condition.simplify_without_ident().unwrap();

    let (input, inner) = context(
        "if block",
        cut(delimited(
            preceded(multispace0, tag("{")),
            take_until_unbalanced('{', '}'),
            tag("}"),
        )),
    )(input)?;

    let (_, inner_tokens) = parse_tokens(inner)?;

    Ok((
        input,
        Token::If {
            tokens: inner_tokens,
            condition,
        },
    ))
}

fn parse_range(input: &str) -> IResult<&str, Range> {
    let (input, exp1) = parse_expr(input)?;
    let (input, ty) = preceded(multispace0, alt((tag(".."), tag("..="))))(input)?;
    let (input, exp2) = parse_expr(input)?;
    Ok((
        input,
        match ty {
            ".." => Range::Exclusive((exp1, exp2)),
            "..=" => Range::Inclusive((exp1, exp2)),
            _ => unreachable!(),
        },
    ))
}

fn parse_for(input: &str) -> IResult<&str, Token> {
    let (input, _) = tag("for")(input)?;

    // let (input, );
    todo!()
}

pub fn parse_tokens(mut input: &str) -> IResult<&str, Vec<Token>> {
    let mut out = Vec::new();

    // Consume initial shader code, up to the first "#"
    let (new_input, initial_token) = parse_shader_code(input)?;
    out.push(initial_token);
    input = new_input;

    while input != "" {
        let (new_input, _) = char('#')(input)?;
        input = new_input;
        // Parse directive
        let (new_input, token) = alt((
            parse_if,
            parse_shader_code
        ))(input)?;
        out.push(token);
        input = new_input;
    }
    Ok((input, out))
}
