from datetime import datetime
import os, sys, re

def tryparse(t, s):
  try:
    return t(s)
  except Exception:
    return None

def parse_from_savename_value(types, string):
  for t in types:
    res = tryparse(t, string)
    if res is not None:
      return res
  return string


def projectdir(*paths):
  return os.path.join(os.path.dirname(sys.prefix), *paths)

def datadir(*paths):
  return projectdir("data", *paths)

def srcdir(*paths):
  return projectdir("src", *paths)

def parse_savename(filename: str, parsetypes = (int, float), connector: str = '_'):
  assert len(connector) == 1
  prefix_part, savename_part = os.path.dirname(filename), os.path.basename(filename)
  # Extract the suffix. A suffix is identified by searching for the last "."
  # after the last "=".
  last_eq = savename_part.rfind('=')
  last_dot = savename_part.rfind('.')
  if (last_dot == -1) or (last_eq > last_dot):
    # if no dot is after the last "="
    # there is no suffix
    name, suffix = savename_part, ''
  else:
    # Check if the last dot is part of a float number by parsing it as Int
    if tryparse(int, savename_part[last_dot+1:]) == None:
      name, suffix = savename_part[:last_dot-1], savename_part[last_dot+1:]
    else:
      name, suffix = savename_part, ''
  first_eq = name.find('=')
  first_connector = name.find(connector)
  if (first_connector == -1) or (first_eq < first_connector):
    prefix, _parameters = '', name
  else:
    prefix, _parameters = name[:first_connector-1], name[first_connector+1:]
  prefix = os.path.join(prefix_part, prefix)
  parameters = {}
  name_separator = f"=[^{connector}]+{connector}"
  c_idx = 0
  for next_range in re.finditer(name_separator, _parameters):
    equal_sign, end_of_value = next_range.span()
    key = _parameters[c_idx:equal_sign]
    val = parse_from_savename_value(parsetypes, _parameters[equal_sign+1:end_of_value-1])
    parameters[key] = val
    c_idx = end_of_value
  _parameters_aux = _parameters[c_idx:]
  equal_sign = _parameters_aux.find('=')
  assert equal_sign != -1, f"Savename cannot be parsed. There is a {connector} after the last '='. Values containing {connector} are note allowed when parsing."
  parameters[_parameters_aux[:equal_sign]] = parse_from_savename_value(parsetypes, _parameters_aux[equal_sign+1:])
  return prefix, parameters, suffix

def valtostring(val, digits):
  if type(val) == float:
    return round(val, ndigits=digits)
  else:
    return str(val)

def savename(d: dict, suffix="", digits=3, sort=True, connector='_', allowedtypes=(int, float, str, datetime)):
  labels = list(d.keys())
  val2string = lambda val: valtostring(val, digits)
  if sort:
    labels.sort()
  s = ""
  for k in labels:
    val = d[k]
    t = type(val)
    if t not in allowedtypes:
      continue
    entry = f"{k}={val2string(val)}"
    if len(s) != 0:
      s += connector
    s += entry
  if suffix != "":
    s += "." + suffix
  return s
