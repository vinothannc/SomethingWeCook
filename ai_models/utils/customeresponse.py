from rest_framework.response import Response


def create_response(result, status: bool, code: int, message=None, extra=None):
    """
    This method is used to create a custom response to return
    on each API call.

    Params:
       result : Json data or data which will be access frontend
       status : True or False
       code : Http response code like 200,400
       message : Custom message to pop up on FE
       extra : Apart from json if any other data you want to Append

    Returns:
        HTTP response 
        
    """
    if not message:
        if code == 400:
            message = "Bad request"
        elif code == 200:
            message = "Success"

    if extra:
        result.update(extra)

    return Response(
        data={"status": status, "message": message, "result": result},
        status=code,
    )
