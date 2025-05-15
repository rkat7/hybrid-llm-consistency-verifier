may(second).
jay(third).
%— Domain rules you add by hand (or extend as needed) —%

% Universal rule: every bird flies, unless exception
flies(X) :- bird(X), not exception(X).

% You can’t be both single and married
:- married(X), single(X).

% You can’t both fly and not fly
:- flies(X), not_flies(X).

% Show only the facts you care about
#show married/1.
#show single/1.
#show flies/1.
#show not_flies/1.
