% Fixed rules based on generated_rules_20250514_235643.lp
% Modified to show inconsistency

% Rule 1: A person cannot be both john and may at the same time.
:- person(X), john(X), may(X).

% Rule 2: A person cannot be both may and jay at the same time.
:- person(X), may(X), jay(X).

% Rule 3: If a person is john, then they must be married.
married(X) :- person(X), john(X).

% Rule 4: If a person is may, then they must be married.
married(X) :- person(X), may(X).

% Rule 5: If a person is jay, then they must be married.
married(X) :- person(X), jay(X).

% Rule 6: When John is the same as May, they become both john and may predicates
may(john) :- same_as(john,may).
john(may) :- same_as(john,may).

% Define all entities as persons
person(john).
person(may).
person(jay).

% Show directives
#show person/1.
#show married/1.
#show john/1.
#show may/1.
#show jay/1.
#show same_as/2. 