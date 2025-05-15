% Auto-generated rules from input.txt
% Generated on: 2025-05-14T23:56:43.519378
% Category: custom_rules

% Rule 1: An entity cannot be both john and may at the same time.
:- john(X), may(X).

% Rule 2: An entity cannot be both may and john at the same time.
:- may(X), john(X).

% Rule 3: An entity cannot be both jay and may at the same time.
:- jay(X), may(X).

% Rule 4: If an entity is john, then it is also jay.
jay(X) :- john(X).

% Rule 5: There must be exactly 4 entities that are may.
:- not { may(X) } = 4.

% Show directives
#show jay/1.
#show may/1.
#show john/1.
