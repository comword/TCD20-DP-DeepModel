import src.gen_proto.student_pb2_grpc as student_pb2__grpc

class StudentAppServicer(student_pb2__grpc.StudentAppServicer):
    def __init__(self):
        pass

    def UpPredictResult(self, request, context):
        pass

    def GetExams(self, request, context):
        pass

    def GetPredicts(self, request, context):
        pass

    def GetUserDetail(self, request, context):
        pass

    def PutUserDetail(self, request, context):
        pass

    def StreamVideo(self, request_iterator, context):
        pass