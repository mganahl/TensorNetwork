# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def get_jax_precision(jax, precision):
  if precision is None:
    return jax.lax.Precision.DEFAULT
  if precision == "DEFAULT":
    return jax.lax.Precision.DEFAULT
  if precision == "HIGH":
    return jax.lax.Precision.HIGH
  if precision == "HIGHEST":
    return jax.lax.Precision.HIGHEST
  raise ValueError(f"unknown value {precision}"
                   f" for precision.")
