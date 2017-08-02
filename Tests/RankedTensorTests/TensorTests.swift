//
//  TensorTests.swift
//  RankedTensor
//
//  Copyright 2016-2017 DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import XCTest
@testable import RankedTensor
import struct CoreTensor.TensorIndex
import struct CoreTensor.TensorShape

class RankedTensorTests: XCTestCase {

    func testInit() {
        let vector = Vector<Int>(shape: (3), repeating: 5)
        XCTAssertTrue(vector.shape == (3))
        XCTAssertEqual(vector.units, ContiguousArray(repeating: 5, count: 3))

        let matrix = Matrix<Int>(shape: (4, 5), repeating: 1)
        XCTAssertTrue(matrix.shape == (4, 5))
        XCTAssertEqual(matrix.units, ContiguousArray(repeating: 1, count: 20))
    }

    func testRanks() {
        let tensor = Tensor3D<Int>(shape: (1, 2, 3), repeating: 0)
        XCTAssertTrue(tensor.shape == (1, 2, 3))
        XCTAssertEqual(tensor.shape.0, 1)
        XCTAssertEqual(tensor.shape.1, 2)
        XCTAssertEqual(tensor.shape.2, 3)
    }

    static var allTests : [(String, (RankedTensorTests) -> () throws -> Void)] {
        return [
            ("testInit", testInit),
            ("testRanks", testRanks),
        ]
    }

}

