"""S3 state tracker for dagger.

The S3 state tracker takes a prefix, and the layout is as such.

<prefix>/
  <instanceid>/
    graph - The graph definition
    params/
      <paramname> - Data passed to parameter storage manager to
                    retrieve the parameter.
    results/ - The results of the outputs of various operations
      <opid> - Data that can be passed into the operations storage
               manager to retrieve the result of the operation
    outputs/ - The outputs of the graph run
      <output> - The result of the operation

"""
